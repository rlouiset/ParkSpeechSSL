from __future__ import annotations

from typing import TypedDict

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import Wav2Vec2Model

from pdspeech_ssl.augment import feature_time_mask
from pdspeech_ssl.config import EncoderHParams, ModelHParams, AugmentHParams


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TemporalTransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model=128, n_heads=1, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.dropout = nn.Dropout(dropout)

        # Learnable [CLS] token (1, 1, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, lengths):
        """
        Args:
            x: (B, T, F)
            lengths: (B,) actual sequence lengths
        Returns:
            cls_out: (B, d_model)
        """
        B, T, _ = x.shape
        device = x.device

        # Project to d_model
        x = self.input_proj(x)
        x = self.dropout(x)

        # Add [CLS] token at the beginning
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls_token, x], dim=1)  # (B, T+1, d_model)

        # Build key padding mask (True = pad)
        mask = torch.arange(T, device=device)[None, :] >= lengths[:, None]
        mask = torch.cat([torch.zeros(B, 1, device=device, dtype=torch.bool), mask], dim=1)  # add CLS=False

        # Transformer encoding
        x = self.encoder(x, src_key_padding_mask=mask)

        # Return only the [CLS] embedding
        cls_out = x[:, 0, :]  # (B, d_model)
        return cls_out


class SSLOutput(TypedDict):
    embd: torch.Tensor  # representation used for the downstream HC/PD linear probe (via .detach())
    proj: torch.Tensor  # SimCLR-style projection-head output, used only for the contrastive loss


def _build_wav2vec2(enc_cfg: EncoderHParams) -> Wav2Vec2Model:
    wav2vec2 = Wav2Vec2Model.from_pretrained(enc_cfg.checkpoint)
    # The CNN feature extractor is always frozen -- standard practice for
    # wav2vec2 fine-tuning regardless of trainable_mode, since it's a low-level
    # filterbank-like front end with little to gain from adaptation on a
    # dataset this size.
    wav2vec2.feature_extractor._freeze_parameters()

    if enc_cfg.trainable_mode == "frozen":
        for p in wav2vec2.parameters():
            p.requires_grad = False
    elif enc_cfg.trainable_mode == "lora":
        for p in wav2vec2.parameters():
            p.requires_grad = False
        lora_cfg = LoraConfig(
            r=enc_cfg.lora_r,
            lora_alpha=enc_cfg.lora_alpha,
            lora_dropout=enc_cfg.lora_dropout,
            target_modules=enc_cfg.lora_target_modules,
            bias="none",
        )
        wav2vec2 = get_peft_model(wav2vec2, lora_cfg)
    elif enc_cfg.trainable_mode == "full":
        pass  # everything but the (already frozen) feature extractor stays trainable
    else:
        raise ValueError(f"Unknown trainable_mode: {enc_cfg.trainable_mode}")

    return wav2vec2


class SSLEncoder(nn.Module):
    def __init__(self, encoder_cfg: EncoderHParams, model_cfg: ModelHParams):
        super().__init__()
        self.model_cfg = model_cfg
        self.wav2vec2 = _build_wav2vec2(encoder_cfg)
        in_dim = self.wav2vec2.config.hidden_size

        self.projector = None
        if model_cfg.d_proj is not None:
            self.projector = nn.Linear(in_dim, model_cfg.d_proj)
        d_temporal_input = model_cfg.d_proj if model_cfg.d_proj is not None else in_dim

        self.temporal_transformer = TemporalTransformerEncoder(
            input_dim=d_temporal_input,
            d_model=model_cfg.d_blstm,
            n_heads=model_cfg.n_att,
            num_layers=model_cfg.blstm_layers,
            dropout=model_cfg.dropout,
        )

        self.embedding_projector = nn.Linear(model_cfg.d_blstm, model_cfg.d_emb)
        self.proj_head = nn.Sequential(
            nn.Linear(model_cfg.d_emb, model_cfg.d_emb),
            nn.ReLU(),
            nn.Linear(model_cfg.d_emb, model_cfg.proj_head_dim),
        )

    def wav2vec2_frame_lengths(self, sample_lengths: torch.Tensor) -> torch.Tensor:
        base_model = self.wav2vec2.get_base_model() if hasattr(self.wav2vec2, "get_base_model") else self.wav2vec2
        return base_model._get_feat_extract_output_lengths(sample_lengths).long()

    def encode_frames(self, waveforms: torch.Tensor, sample_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Runs wav2vec2 and returns (frame_features (B,T,H), frame_lengths (B,))."""
        B, T = waveforms.shape
        device = waveforms.device
        attention_mask = torch.arange(T, device=device)[None, :] < sample_lengths[:, None]

        no_grad_wav2vec2 = next(self.wav2vec2.parameters()).requires_grad is False and not any(
            p.requires_grad for p in self.wav2vec2.parameters()
        )
        ctx = torch.no_grad() if no_grad_wav2vec2 else torch.enable_grad()
        with ctx:
            out = self.wav2vec2(waveforms, attention_mask=attention_mask.long())
        frame_features = out.last_hidden_state  # (B, T', H)
        frame_lengths = self.wav2vec2_frame_lengths(sample_lengths)
        frame_lengths = frame_lengths.clamp(max=frame_features.shape[1])
        return frame_features, frame_lengths

    def forward(
        self,
        waveforms: torch.Tensor,
        sample_lengths: torch.Tensor,
        augment_cfg: AugmentHParams | None = None,
    ) -> SSLOutput:
        X, lengths = self.encode_frames(waveforms, sample_lengths)

        if augment_cfg is not None and self.training and torch.rand(()) < augment_cfg.feature_mask_prob:
            X = feature_time_mask(X, lengths, augment_cfg.feature_mask_fraction)

        if self.projector is not None:
            X = self.projector(X)

        H_speech = self.temporal_transformer(X, lengths)
        embd = self.embedding_projector(H_speech)
        proj = self.proj_head(embd)

        return {"embd": embd, "proj": proj}
