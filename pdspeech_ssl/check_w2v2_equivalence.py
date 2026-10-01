"""Checks the memory-efficient quantizer / masked-frame w2v2 loss in model.py against
HF's original code paths (same weights, same masks/negatives/RNG), on GPU under bf16
autocast like training, then in full fp32 (to separate real differences from bf16
rounding amplified through the backbone). Run on a GPU node:
    python3 -m pdspeech_ssl.check_w2v2_equivalence <wav2vec2 checkpoint path>
"""
from __future__ import annotations

import sys

import torch
from transformers import Wav2Vec2ForPreTraining
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2GumbelVectorQuantizer,
    _compute_mask_indices,
    _sample_negative_indices,
)

from pdspeech_ssl.model import _w2v2_pretraining_loss


def _set_precision(precision: str) -> None:
    fp32 = precision == "fp32"
    torch.set_float32_matmul_precision("highest" if fp32 else "medium")
    torch.backends.cuda.matmul.allow_tf32 = not fp32
    torch.backends.cudnn.allow_tf32 = not fp32


def _run(model, wav, attn, mask, negatives, quantizer_cls, ours: bool, precision: str):
    model.quantizer.__class__ = quantizer_cls
    model.zero_grad()
    torch.manual_seed(0)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
        if ours:
            out = model(wav, attention_mask=attn, mask_time_indices=mask, return_dict=True)
            contrastive, diversity = _w2v2_pretraining_loss(
                model.config, out.projected_states, out.projected_quantized_states,
                out.codevector_perplexity, mask, negatives,
            )
        else:
            out = model(wav, attention_mask=attn, mask_time_indices=mask,
                        sampled_negative_indices=negatives, return_dict=True)
            contrastive, diversity = out.contrastive_loss, out.diversity_loss
    # gradients w.r.t. the loss's own inputs: compares the two losses as functions,
    # before anything is propagated back through the 24-layer backbone
    out.projected_states.retain_grad()
    out.projected_quantized_states.retain_grad()
    loss = contrastive + model.config.diversity_loss_weight * diversity
    loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    grads["<loss input> projected_states"] = out.projected_states.grad.clone()
    grads["<loss input> projected_quantized_states"] = out.projected_quantized_states.grad.clone()
    return contrastive.detach(), diversity.detach(), grads


def main(path: str) -> None:
    model = Wav2Vec2ForPreTraining.from_pretrained(
        path, local_files_only=True, layerdrop=0.0, apply_spec_augment=True, num_negatives=100
    ).cuda()
    model.config.mask_time_prob = 0.0
    model.config.mask_feature_prob = 0.0
    model.freeze_feature_encoder()
    # eval: no dropout, so both runs see the same graph (explicit masks still apply);
    # the quantizer's own train/eval mode is toggled per case below
    model.eval()

    torch.manual_seed(0)
    lengths = torch.tensor([16000 * 6, 16000 * 4, 16000 * 5])
    wav = torch.randn(3, int(lengths.max()), device="cuda")
    attn = (torch.arange(wav.shape[1])[None] < lengths[:, None]).long().cuda()
    num_frames = int(model._get_feat_extract_output_lengths(torch.tensor(wav.shape[1])))
    sub_attn = model._get_feature_vector_attention_mask(num_frames, attn)
    mask_np = _compute_mask_indices((3, num_frames), 0.65, 10, attention_mask=sub_attn, min_masks=2)
    neg_np = _sample_negative_indices((3, num_frames), 100, mask_time_indices=mask_np)
    mask = torch.from_numpy(mask_np).cuda().bool()
    negatives = torch.from_numpy(neg_np).cuda().long()

    def rel_diffs(a, b):
        # per-param ||ga - gb|| / ||ga||, worst first. Skips attention k_proj biases: their
        # true gradient is exactly 0 (a shared key offset shifts every score in a softmax
        # row by the same q.b), so what's measured there is pure rounding noise.
        return sorted(
            (((a[n] - b[n]).float().norm() / a[n].float().norm().clamp_min(1e-30)).item(), n,
             a[n].float().norm().item())
            for n in a
            if not n.endswith("k_proj.bias")
        )[::-1]

    # the quantizer already matched the noise baseline in both modes; this isolates the loss
    model.quantizer.eval()
    HFQ = Wav2Vec2GumbelVectorQuantizer
    for precision in ("bf16", "fp32"):
        _set_precision(precision)
        ref = _run(model, wav, attn, mask, negatives, HFQ, ours=False, precision=precision)
        runs = {
            "HF rerun (noise baseline)": _run(model, wav, attn, mask, negatives, HFQ, ours=False, precision=precision),
            "our loss": _run(model, wav, attn, mask, negatives, HFQ, ours=True, precision=precision),
        }
        print(f"\n=== {precision} ===")
        for name, new in runs.items():
            diffs = rel_diffs(ref[2], new[2])
            param_diffs = sorted(d for d, n, _ in diffs if not n.startswith("<"))
            print(
                f"{name:28s} contrastive {ref[0].item():.6f} vs {new[0].item():.6f} | "
                f"diversity {ref[1].item():.6f} vs {new[1].item():.6f} | "
                f"same grad params: {ref[2].keys() == new[2].keys()} | "
                f"median param rel grad diff {param_diffs[len(param_diffs) // 2]:.2e}"
            )
            for d, n, norm in diffs:
                if n.startswith("<"):
                    print(f"    rel grad diff {d:.2e}  (|g_ref|={norm:.3e})  {n}")
            for d, n, norm in [x for x in diffs if not x[1].startswith("<")][:3]:
                print(f"    rel grad diff {d:.2e}  (|g_ref|={norm:.3e})  {n}")

if __name__ == "__main__":
    main(sys.argv[1])
