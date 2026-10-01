from __future__ import annotations

from typing import TypedDict

import torch
import torch.nn as nn
from transformers import Wav2Vec2ForPreTraining
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2GumbelVectorQuantizer,
    _compute_mask_indices,
    _sample_negative_indices,
)

from pdspeech_ssl.augment import feature_time_mask
from pdspeech_ssl.config import AugmentHParams, EncoderHParams, ModelHParams, W2V2HParams

TRAINABLE_MODES = ("frozen", "full")
LAYER_AGGREGATIONS = ("last", "weighted")
# heads Wav2Vec2ForPreTraining needs for the w2v2 loss -- must come from the checkpoint,
# a randomly initialized quantizer/project_q would make the targets meaningless.
_PRETRAINING_HEAD_PREFIXES = ("quantizer", "project_q", "project_hid")


class _SSLOutputBase(TypedDict):
    embd: torch.Tensor  # (B, d_emb)
    proj: torch.Tensor  # (B, proj_head_dim)


class SSLOutput(_SSLOutputBase, total=False):
    # only present for a w2v2_mask=True forward; losses are per masked frame
    w2v2_loss: torch.Tensor  # contrastive + diversity_loss_weight * diversity
    w2v2_contrastive: torch.Tensor
    w2v2_diversity: torch.Tensor
    codevector_perplexity: torch.Tensor


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


class _MemoryEfficientGumbelQuantizer(Wav2Vec2GumbelVectorQuantizer):
    """Same outputs as HF's quantizer, without its dense (B*T, G*V, D/G) intermediate
    (`codevector_probs.unsqueeze(-1) * self.codevectors`, ~1 MB/frame for XLSR-53:
    ~14 GiB for a 16 x 10s batch). Eval: exact gather of the argmax codevectors.
    Train: HF's exact ops (no matmul, so unaffected by float32_matmul_precision),
    just run over chunks of frames to cap the intermediate at ~0.5 GiB."""

    _CHUNK_FRAMES = 512

    def forward(self, hidden_states: torch.Tensor, mask_time_indices: torch.Tensor | None = None):
        batch_size, sequence_length, _ = hidden_states.shape
        G, V = self.num_groups, self.num_vars
        logits = self.weight_proj(hidden_states).view(batch_size * sequence_length, G, V)
        codebook = self.codevectors.view(G, V, -1)  # (G, V, D/G)

        if self.training:
            codevector_probs = nn.functional.gumbel_softmax(logits.float(), tau=self.temperature, hard=True).type_as(logits)
            perplexity = self._compute_perplexity(torch.softmax(logits.float(), dim=-1), mask_time_indices)
            codevectors = torch.cat([
                (p.reshape(p.shape[0], -1).unsqueeze(-1) * self.codevectors).view(p.shape[0], G, V, -1).sum(-2)
                for p in codevector_probs.split(self._CHUNK_FRAMES)
            ])
        else:
            codevector_idx = logits.argmax(dim=-1)  # (N, G)
            codevector_probs = torch.zeros_like(logits).scatter_(-1, codevector_idx.unsqueeze(-1), 1.0)
            perplexity = self._compute_perplexity(codevector_probs, mask_time_indices)
            codevectors = codebook[torch.arange(G, device=logits.device), codevector_idx]  # (N, G, D/G)

        return codevectors.reshape(batch_size, sequence_length, -1), perplexity


def _build_wav2vec2(enc_cfg: EncoderHParams, w2v2_cfg: W2V2HParams) -> Wav2Vec2ForPreTraining:
    if enc_cfg.trainable_mode not in TRAINABLE_MODES:
        raise ValueError(f"Unknown trainable_mode: {enc_cfg.trainable_mode!r}, expected one of {TRAINABLE_MODES}")

    model, loading_info = Wav2Vec2ForPreTraining.from_pretrained(
        enc_cfg.model_name_or_path,
        output_loading_info=True,
        local_files_only=True,
        layerdrop=0.0,
        # REQUIRED: if False, HF's _mask_hidden_states also ignores the explicit mask_time_indices
        apply_spec_augment=True,
        num_negatives=w2v2_cfg.num_negatives,
        diversity_loss_weight=w2v2_cfg.diversity_loss_weight,
    )
    missing_heads = [k for k in loading_info["missing_keys"] if k.startswith(_PRETRAINING_HEAD_PREFIXES)]
    if missing_heads:
        raise ValueError(
            f"{enc_cfg.model_name_or_path} is missing wav2vec2 pretraining weights {missing_heads} -- "
            "it must be a pretraining checkpoint (e.g. wav2vec2-large-xlsr-53), not a fine-tuned one."
        )
    if not hasattr(model.wav2vec2, "masked_spec_embed"):
        raise ValueError(
            f"{enc_cfg.model_name_or_path}'s config has mask_time_prob == mask_feature_prob == 0, so HF "
            "never built masked_spec_embed, which the explicit-mask w2v2 forward needs."
        )
    # No HF-internal random masking: masks are passed explicitly (SSLEncoder.forward).
    # Zeroed here, *after* loading, rather than passed as from_pretrained overrides:
    # Wav2Vec2Model only creates masked_spec_embed when one of them is > 0 at init, so
    # overriding them to 0 at load time would drop the pretrained mask embedding.
    model.config.mask_time_prob = 0.0
    model.config.mask_feature_prob = 0.0
    # same module/weights, memory-efficient forward (see _MemoryEfficientGumbelQuantizer)
    model.quantizer.__class__ = _MemoryEfficientGumbelQuantizer

    if enc_cfg.trainable_mode == "frozen":
        for p in model.parameters():
            p.requires_grad = False
    else:  # "full"
        for p in model.parameters():
            p.requires_grad = True
        model.freeze_feature_encoder()  # CNN feature encoder always frozen
        if w2v2_cfg.train_quantizer:
            model.set_gumbel_temperature(w2v2_cfg.gumbel_temperature)
        else:
            for p in [*model.quantizer.parameters(), *model.project_q.parameters()]:
                p.requires_grad = False

    return model


class SSLEncoder(nn.Module):

    def __init__(
        self,
        encoder_cfg: EncoderHParams,
        model_cfg: ModelHParams,
        w2v2_cfg: W2V2HParams,
        sample_rate: int = 16_000,
    ):
        super().__init__()
        if model_cfg.layer_aggregation not in LAYER_AGGREGATIONS:
            raise ValueError(
                f"Unknown model.layer_aggregation: {model_cfg.layer_aggregation!r}, expected one of {LAYER_AGGREGATIONS}"
            )

        self.model_cfg = model_cfg
        self.w2v2_cfg = w2v2_cfg
        self.trainable_mode = encoder_cfg.trainable_mode
        self.sample_rate = sample_rate

        # HF model kept whole: .wav2vec2 (inner encoder), .quantizer, .project_q/.project_hid,
        # plus the length/attention-mask helpers used in forward.
        self._w2v2_base = _build_wav2vec2(encoder_cfg, w2v2_cfg)
        w2v2_config = self._w2v2_base.config
        wav2vec_dim = w2v2_config.hidden_size

        # one weight per hidden state (feature projection output + each Transformer layer);
        # zeros => uniform softmax at init
        self.layer_weights = (
            nn.Parameter(torch.zeros(w2v2_config.num_hidden_layers + 1))
            if model_cfg.layer_aggregation == "weighted"
            else None
        )

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

    def train(self, mode: bool = True) -> "SSLEncoder":
        super().train(mode)
        if self.trainable_mode == "frozen":
            self._w2v2_base.eval()  # no dropout in a backbone that isn't being trained
        elif not self.w2v2_cfg.train_quantizer:
            # HF's quantizer samples with Gumbel noise whenever it's in train mode;
            # eval mode => argmax codevectors, i.e. deterministic w2v2 targets.
            self._w2v2_base.quantizer.eval()
        return self

    def layer_weight_probs(self) -> torch.Tensor | None:
        return torch.softmax(self.layer_weights, dim=0) if self.layer_weights is not None else None

    @staticmethod
    def _normalize_waveforms(
        waveforms: torch.Tensor,
        sample_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched, on-device equivalent of Wav2Vec2FeatureExtractor(do_normalize=True):
        per-utterance zero-mean/unit-variance over the valid samples only, padding
        zeroed. Returns (input_values (B, L), attention_mask (B, L) long)."""
        L = waveforms.shape[1]
        sample_lengths = sample_lengths.to(waveforms.device)
        attention_mask = (torch.arange(L, device=waveforms.device)[None, :] < sample_lengths[:, None]).long()

        valid = attention_mask.to(torch.float32)
        x = waveforms.to(torch.float32)
        n = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (x * valid).sum(dim=1, keepdim=True) / n
        var = (((x - mean) * valid) ** 2).sum(dim=1, keepdim=True) / n
        x = (x - mean) / torch.sqrt(var + 1e-7) * valid
        return x, attention_mask

    def _w2v2_masked_forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], dict[str, torch.Tensor]]:
        """Wav2Vec2ForPreTraining forward with explicit span masks + within-utterance
        negatives. Returns (hidden_states, w2v2 loss terms normalized per masked frame)."""
        batch_size = input_values.shape[0]
        num_frames = int(
            self._w2v2_base._get_feat_extract_output_lengths(torch.tensor(input_values.shape[1]))
        )
        sub_attention_mask = self._w2v2_base._get_feature_vector_attention_mask(num_frames, attention_mask)

        mask_np = _compute_mask_indices(
            (batch_size, num_frames),
            mask_prob=self.w2v2_cfg.mask_prob,
            mask_length=self.w2v2_cfg.mask_length,
            attention_mask=sub_attention_mask,
            min_masks=self.w2v2_cfg.min_masks,
        )
        negatives_np = _sample_negative_indices(
            (batch_size, num_frames), self.w2v2_cfg.num_negatives, mask_time_indices=mask_np
        )
        mask = torch.from_numpy(mask_np).to(device=input_values.device, dtype=torch.bool)
        negatives = torch.from_numpy(negatives_np).to(device=input_values.device, dtype=torch.long)

        out = self._w2v2_base(
            input_values,
            attention_mask=attention_mask,
            mask_time_indices=mask,
            sampled_negative_indices=negatives,
            output_hidden_states=True,
            return_dict=True,
        )
        # HF sums the contrastive/diversity losses over masked frames
        n = mask.sum()
        losses = {
            "w2v2_loss": out.loss / n,
            "w2v2_contrastive": out.contrastive_loss / n,
            "w2v2_diversity": out.diversity_loss / n,
            "codevector_perplexity": out.codevector_perplexity,
        }
        return out.hidden_states, losses

    def forward(
        self,
        waveforms: torch.Tensor,
        sample_lengths: torch.Tensor,
        augment_cfg: AugmentHParams | None = None,
        w2v2_mask: bool = False,
    ) -> SSLOutput:
        input_values, attention_mask = self._normalize_waveforms(waveforms, sample_lengths)
        lengths = self._w2v2_base._get_feat_extract_output_lengths(attention_mask.sum(dim=-1)).long()

        out: SSLOutput = {}
        if w2v2_mask:
            hidden_states, w2v2_losses = self._w2v2_masked_forward(input_values, attention_mask)
            out.update(w2v2_losses)
        else:
            backbone_trainable = any(p.requires_grad for p in self._w2v2_base.wav2vec2.parameters())
            with torch.set_grad_enabled(torch.is_grad_enabled() and backbone_trainable):
                hidden_states = self._w2v2_base.wav2vec2(
                    input_values,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                ).hidden_states

        if self.layer_weights is not None:
            weights = self.layer_weight_probs()
            X = (weights[:, None, None, None] * torch.stack(hidden_states, dim=0)).sum(dim=0)  # (B, T, H)
        else:
            X = hidden_states[-1]  # (B, T, H)

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

        out["embd"] = embd
        out["proj"] = proj
        return out
