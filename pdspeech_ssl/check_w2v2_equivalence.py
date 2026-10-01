"""Checks the memory-efficient quantizer / masked-frame w2v2 loss in model.py against
HF's original code paths (same weights, same masks/negatives/RNG), on GPU under bf16
autocast like training. Run on a GPU node:
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

from pdspeech_ssl.model import _MemoryEfficientGumbelQuantizer, _w2v2_pretraining_loss

torch.set_float32_matmul_precision("medium")


def _run(model, wav, attn, mask, negatives, quantizer_cls, ours: bool):
    model.quantizer.__class__ = quantizer_cls
    model.zero_grad()
    torch.manual_seed(0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
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
    loss = contrastive + model.config.diversity_loss_weight * diversity
    loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
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
        # per-param ||ga - gb|| / ||ga||, worst first
        return sorted(
            (((a[n] - b[n]).float().norm() / a[n].float().norm().clamp_min(1e-30)).item(), n,
             a[n].float().norm().item())
            for n in a
        )[::-1]

    HFQ, MyQ = Wav2Vec2GumbelVectorQuantizer, _MemoryEfficientGumbelQuantizer
    for train_quantizer in (False, True):
        model.quantizer.train(train_quantizer)
        mode = "train" if train_quantizer else "eval"
        ref = _run(model, wav, attn, mask, negatives, HFQ, ours=False)
        runs = {
            "HF rerun (noise baseline)": _run(model, wav, attn, mask, negatives, HFQ, ours=False),
            "our quantizer + HF loss": _run(model, wav, attn, mask, negatives, MyQ, ours=False),
            "HF quantizer + our loss": _run(model, wav, attn, mask, negatives, HFQ, ours=True),
            "ours (both)": _run(model, wav, attn, mask, negatives, MyQ, ours=True),
        }
        print(f"\n=== quantizer {mode} ===")
        for name, new in runs.items():
            worst = rel_diffs(ref[2], new[2])
            print(
                f"{name:28s} contrastive {ref[0].item():.6f} vs {new[0].item():.6f} | "
                f"diversity {ref[1].item():.6f} vs {new[1].item():.6f} | "
                f"same grad params: {ref[2].keys() == new[2].keys()}"
            )
            for d, n, norm in worst[:3]:
                print(f"    rel grad diff {d:.2e}  (|g_ref|={norm:.3e})  {n}")


if __name__ == "__main__":
    main(sys.argv[1])
