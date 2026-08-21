from __future__ import annotations

from typing import TypedDict

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
)

from pdspeech_ssl.augment import feature_time_mask
from pdspeech_ssl.config import (
    AugmentHParams,
    EncoderHParams,
    ModelHParams,
)


# ---------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------

class SSLOutput(TypedDict):
    """
    embd:
        Utterance-level representation used for downstream tasks.
        Shape: (B, d_emb)

    proj:
        Projection-head representation used for contrastive learning.
        Shape: (B, proj_head_dim)
    """

    embd: torch.Tensor
    proj: torch.Tensor


# ---------------------------------------------------------------------
# Self-attention pooling
# ---------------------------------------------------------------------

class SelfAttentionPooling(nn.Module):
    """
    Learned attention pooling over the temporal dimension.

    Input:
        batch_rep: (B, T, H)
        lengths:   (B,)

    Output:
        utter_rep: (B, H)

    Padding frames are excluded from the attention computation.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.W = nn.Linear(input_dim, 1)

    def forward(
        self,
        batch_rep: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:

        # batch_rep: (B, T, H)
        B, T, _ = batch_rep.shape

        # True for valid frames, False for padding
        valid_mask = (
            torch.arange(
                T,
                device=batch_rep.device,
            )[None, :]
            < lengths[:, None]
        )

        # Attention scores: (B, T)
        scores = self.W(batch_rep).squeeze(-1)

        # Prevent attention to padding
        scores = scores.masked_fill(
            ~valid_mask,
            float("-inf"),
        )

        # Attention weights: (B, T)
        weights = torch.softmax(scores, dim=1)

        # Weighted temporal average:
        #
        # (B, T, 1) * (B, T, H)
        # -> (B, T, H)
        # -> sum over T
        # -> (B, H)
        utter_rep = torch.sum(
            batch_rep * weights.unsqueeze(-1),
            dim=1,
        )

        return utter_rep


# ---------------------------------------------------------------------
# BLSTM
# ---------------------------------------------------------------------

class BiLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM over Wav2Vec2 frame features.

    Input:
        x:
            (B, T, F)

        lengths:
            (B,)

    Output:
        H:
            (B, T, 2*d_model)

    The temporal pooling is deliberately NOT performed here.
    """

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

        # x: (B, T, F)
        B, T, _ = x.shape

        x = self.dropout(x)

        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_out, _ = self.blstm(packed)

        H, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=T,
        )

        # H: (B, T, 2*d_model)
        return H


# ---------------------------------------------------------------------
# Wav2Vec2
# ---------------------------------------------------------------------

def _build_wav2vec2(
    enc_cfg: EncoderHParams,
) -> Wav2Vec2Model:

    model_name = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"

    model = Wav2Vec2Model.from_pretrained(
        model_name,
        output_hidden_states=True,
    )

    # ---------------------------------------------------------------
    # Freeze / LoRA / full fine-tuning
    # ---------------------------------------------------------------

    if enc_cfg.trainable_mode == "frozen":

        for p in model.parameters():
            p.requires_grad = False

    elif enc_cfg.trainable_mode == "lora":

        # Start by freezing the complete Wav2Vec2 model.
        for p in model.parameters():
            p.requires_grad = False

        lora_cfg = LoraConfig(
            r=enc_cfg.lora_r,
            lora_alpha=enc_cfg.lora_alpha,
            lora_dropout=enc_cfg.lora_dropout,
            target_modules=enc_cfg.lora_target_modules,
            bias="none",
        )

        model = get_peft_model(
            model,
            lora_cfg,
        )

        # Useful to verify what is actually trainable.
        model.print_trainable_parameters()

    elif enc_cfg.trainable_mode == "full":

        # Full Wav2Vec2 fine-tuning.
        for p in model.parameters():
            p.requires_grad = True

    else:
        raise ValueError(
            f"Unknown trainable_mode: {enc_cfg.trainable_mode}"
        )

    return model


# ---------------------------------------------------------------------
# SSL encoder
# ---------------------------------------------------------------------

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

        # -------------------------------------------------------------
        # Wav2Vec2 model
        # -------------------------------------------------------------

        self.wav2vec2 = _build_wav2vec2(
            encoder_cfg
        )

        # Wav2Vec2 hidden dimension.
        #
        # For XLSR-53 this is typically 1024.
        #
        # Wav2Vec2 output:
        #     (B, T_frames, hidden_size)
        # -------------------------------------------------------------

        wav2vec_dim = self.wav2vec2.config.hidden_size

        # -------------------------------------------------------------
        # Optional frame-level projection
        # -------------------------------------------------------------

        self.projector = None

        if model_cfg.d_proj is not None:
            self.projector = nn.Linear(
                wav2vec_dim,
                model_cfg.d_proj,
            )

            temporal_input_dim = model_cfg.d_proj

        else:
            temporal_input_dim = wav2vec_dim

        # -------------------------------------------------------------
        # BLSTM
        # -------------------------------------------------------------

        self.temporal_encoder = BiLSTMEncoder(
            input_dim=temporal_input_dim,
            d_model=model_cfg.d_blstm,
            num_layers=model_cfg.blstm_layers,
            dropout=model_cfg.dropout,
        )

        # BLSTM is bidirectional, therefore:
        #
        #     BLSTM output = (B, T, 2*d_blstm)
        #
        # Attention pooling therefore receives 2*d_blstm.
        # -------------------------------------------------------------

        self.self_attention = SelfAttentionPooling(
            2 * model_cfg.d_blstm
        )

        # -------------------------------------------------------------
        # Utterance-level embedding
        # -------------------------------------------------------------

        self.embedding_projector = nn.Linear(
            2 * model_cfg.d_blstm,
            model_cfg.d_emb,
        )

        # -------------------------------------------------------------
        # Contrastive projection head
        # -------------------------------------------------------------

        self.proj_head = nn.Sequential(
            nn.Linear(
                model_cfg.d_emb,
                model_cfg.d_emb,
            ),
            nn.ReLU(),
            nn.Linear(
                model_cfg.d_emb,
                model_cfg.proj_head_dim,
            ),
        )

    # -----------------------------------------------------------------
    # Wav2Vec2 frame lengths
    # -----------------------------------------------------------------

    def wav2vec2_frame_lengths(
        self,
        sample_lengths: torch.Tensor,
    ) -> torch.Tensor:

        """
        Convert waveform lengths in samples into Wav2Vec2
        output lengths in frames.

        Input:
            sample_lengths: (B,)

        Output:
            frame_lengths: (B,)
        """

        base_model = (
            self.wav2vec2.get_base_model()
            if hasattr(self.wav2vec2, "get_base_model")
            else self.wav2vec2
        )

        frame_lengths = (
            base_model
            ._get_feat_extract_output_lengths(
                sample_lengths
            )
            .long()
        )

        return frame_lengths

    # -----------------------------------------------------------------
    # Wav2Vec2 feature extraction
    # -----------------------------------------------------------------

    def encode_frames(
            self,
            waveforms: torch.Tensor,
            sample_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        features = []

        for wav, length in zip(waveforms, sample_lengths):
            wav = wav[:length].detach().cpu().numpy()

            inputs = self.processor(
                wav,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
            )

            input_values = inputs.input_values.to(waveforms.device)

            attention_mask = None
            if hasattr(inputs, "attention_mask") and inputs.attention_mask is not None:
                attention_mask = inputs.attention_mask.to(waveforms.device)

            outputs = self.wav2vec2(
                input_values=input_values,
                attention_mask=attention_mask,
            )

            features.append(outputs.last_hidden_state.squeeze(0))

        # features: list of (T_i, H)
        frame_lengths = torch.tensor(
            [x.shape[0] for x in features],
            device=waveforms.device,
            dtype=torch.long,
        )

        frame_features = nn.utils.rnn.pad_sequence(
            features,
            batch_first=True,
        )

        return frame_features, frame_lengths

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def forward(
        self,
        waveforms: torch.Tensor,
        sample_lengths: torch.Tensor,
        augment_cfg: AugmentHParams | None = None,
    ) -> SSLOutput:

        # =============================================================
        # 1. Wav2Vec2
        # =============================================================
        #
        # waveforms:
        #     (B, T_samples)
        #
        # X:
        #     (B, T_frames, H)
        #
        # =============================================================

        X, lengths = self.encode_frames(
            waveforms,
            sample_lengths,
        )

        # =============================================================
        # 2. Feature-level temporal masking
        # =============================================================

        if (
            augment_cfg is not None
            and self.training
            and torch.rand(
                (),
                device=X.device,
            ) < augment_cfg.feature_mask_prob
        ):
            X = feature_time_mask(
                X,
                lengths,
                augment_cfg.feature_mask_fraction,
            )

        # =============================================================
        # 3. Optional frame-level projection
        # =============================================================
        #
        # (B, T_frames, H)
        # ->
        # (B, T_frames, d_proj)
        #
        # =============================================================

        if self.projector is not None:
            X = self.projector(X)

        # =============================================================
        # 4. BLSTM
        # =============================================================
        #
        # (B, T_frames, d_proj)
        # ->
        # (B, T_frames, 2*d_blstm)
        #
        # =============================================================

        X = self.temporal_encoder(
            X,
            lengths,
        )

        # =============================================================
        # 5. Self-attention pooling
        # =============================================================
        #
        # (B, T_frames, 2*d_blstm)
        # ->
        # (B, 2*d_blstm)
        #
        # =============================================================

        H_speech = self.self_attention(
            X,
            lengths,
        )

        # =============================================================
        # 6. Utterance embedding
        # =============================================================
        #
        # (B, 2*d_blstm)
        # ->
        # (B, d_emb)
        #
        # =============================================================

        embd = self.embedding_projector(
            H_speech
        )

        # =============================================================
        # 7. Contrastive projection head
        # =============================================================
        #
        # (B, d_emb)
        # ->
        # (B, proj_head_dim)
        #
        # =============================================================

        proj = self.proj_head(
            embd
        )

        return {
            "embd": embd,
            "proj": proj,
        }