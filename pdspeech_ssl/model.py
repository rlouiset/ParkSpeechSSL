from __future__ import annotations

from typing import TypedDict

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from pdspeech_ssl.augment import feature_time_mask
from pdspeech_ssl.config import AugmentHParams, EncoderHParams, ModelHParams


class SSLOutput(TypedDict):
    embd: torch.Tensor  # (B, d_emb)
    proj: torch.Tensor  # (B, proj_head_dim)


class SelfAttentionPooling(nn.Module):
    """Learned attention pooling over time."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.W = nn.Linear(input_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        # x: (B, T, H)
        T = x.shape[1]
        mask = torch.arange(T, device=x.device)[None, :] < lengths[:, None]

        scores = self.W(x).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)

        return torch.sum(x * weights.unsqueeze(-1), dim=1)  # (B, H)


class BiLSTMEncoder(nn.Module):
    """Bidirectional LSTM over temporal features."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.blstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        T = x.shape[1]
        x = self.dropout(x)

        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.blstm(packed)

        x, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=T,
        )
        return x  # (B, T, 2*d_model)


def _build_wav2vec2(
    enc_cfg: EncoderHParams,
) -> tuple[Wav2Vec2Model, Wav2Vec2FeatureExtractor]:

    model_name = (
        "/lustre/fswork/projects/rech/haj/uik24xv/"
        "huggingface/wav2vec2-xlsr-53-espeak-cv-ft"
    )

    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_name,
        local_files_only=True,
    )

    model = Wav2Vec2Model.from_pretrained(
        model_name,
        output_hidden_states=True,
        local_files_only=True,
    )

    if enc_cfg.trainable_mode == "frozen":
        for p in model.parameters():
            p.requires_grad = False

    elif enc_cfg.trainable_mode == "lora":
        for p in model.parameters():
            p.requires_grad = False

        lora_cfg = LoraConfig(
            r=enc_cfg.lora_r,
            lora_alpha=enc_cfg.lora_alpha,
            lora_dropout=enc_cfg.lora_dropout,
            target_modules=enc_cfg.lora_target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    elif enc_cfg.trainable_mode == "full":
        for p in model.parameters():
            p.requires_grad = True

    else:
        raise ValueError(
            f"Unknown trainable_mode: {enc_cfg.trainable_mode}"
        )

    return model, processor


class SSLEncoder(nn.Module):

    def __init__(
        self,
        encoder_cfg: EncoderHParams,
        model_cfg: ModelHParams,
        sample_rate: int = 16_000,
    ):
        super().__init__()

        self.model_cfg = model_cfg
        self.sample_rate = sample_rate

        self.wav2vec2, self.processor = _build_wav2vec2(encoder_cfg)
        wav2vec_dim = self.wav2vec2.config.hidden_size

        self.projector = (
            nn.Linear(wav2vec_dim, model_cfg.d_proj)
            if model_cfg.d_proj is not None
            else None
        )
        temporal_input_dim = (
            model_cfg.d_proj
            if model_cfg.d_proj is not None
            else wav2vec_dim
        )

        self.temporal_encoder = BiLSTMEncoder(
            input_dim=temporal_input_dim,
            d_model=model_cfg.d_blstm,
            num_layers=model_cfg.blstm_layers,
            dropout=model_cfg.dropout,
        )

        self.self_attention = SelfAttentionPooling(
            2 * model_cfg.d_blstm
        )

        self.embedding_projector = nn.Linear(
            2 * model_cfg.d_blstm,
            model_cfg.d_emb,
        )

        self.proj_head = nn.Sequential(
            nn.Linear(model_cfg.d_emb, model_cfg.d_emb),
            nn.ReLU(),
            nn.Linear(model_cfg.d_emb, model_cfg.proj_head_dim),
        )

    def encode_frames(
        self,
        waveforms: torch.Tensor,
        sample_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return Wav2Vec2 features as (B, T_frames, H) and lengths."""

        features = []

        for wav, length in zip(waveforms, sample_lengths):
            wav = wav[: int(length)].detach().cpu().numpy()

            inputs = self.processor(
                wav,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
            )

            input_values = inputs.input_values.to(waveforms.device)

            attention_mask = getattr(inputs, "attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(waveforms.device)

            with torch.set_grad_enabled(
                any(p.requires_grad for p in self.wav2vec2.parameters())
            ):
                outputs = self.wav2vec2(
                    input_values=input_values,
                    attention_mask=attention_mask,
                )

            features.append(outputs.last_hidden_state.squeeze(0))

        lengths = torch.tensor(
            [x.shape[0] for x in features],
            dtype=torch.long,
            device=waveforms.device,
        )

        features = nn.utils.rnn.pad_sequence(
            features,
            batch_first=True,
        )

        return features, lengths

    def forward(
        self,
        waveforms: torch.Tensor,
        sample_lengths: torch.Tensor,
        augment_cfg: AugmentHParams | None = None,
    ) -> SSLOutput:

        X, lengths = self.encode_frames(
            waveforms,
            sample_lengths,
        )  # (B, T, H)

        if (
            augment_cfg is not None
            and self.training
            and torch.rand((), device=X.device)
            < augment_cfg.feature_mask_prob
        ):
            X = feature_time_mask(
                X,
                lengths,
                augment_cfg.feature_mask_fraction,
            )

        if self.projector is not None:
            X = self.projector(X)  # (B, T, d_proj)

        X = self.temporal_encoder(
            X,
            lengths,
        )  # (B, T, 2*d_blstm)

        H_speech = self.self_attention(
            X,
            lengths,
        )  # (B, 2*d_blstm)

        embd = self.embedding_projector(
            H_speech
        )  # (B, d_emb)

        proj = self.proj_head(
            embd
        )  # (B, proj_head_dim)

        return {
            "embd": embd,
            "proj": proj,
        }