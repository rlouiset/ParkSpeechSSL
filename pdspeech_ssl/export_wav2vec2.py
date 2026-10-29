"""
Export the (fine-tuned) wav2vec2 of a training checkpoint as a plain HuggingFace folder, loadable
anywhere with Wav2Vec2Model.from_pretrained(out_dir) / Wav2Vec2FeatureExtractor.from_pretrained(out_dir).
Its last_hidden_state (== hidden_states[-1]) is exactly the frame-level feature the BiLSTM was
trained on.

Checkpoints only hold the trainable weights (see SSLLightningModule.on_save_checkpoint), so the
base checkpoint the run started from is loaded first and the checkpoint's model.wav2vec2.*
weights go on top. Anything the run kept frozen (the CNN feature encoder with
encoder.freeze_feature_encoder=true) is then the base model's, as it was during training.
Only "full" runs are supported: a "frozen" run has nothing to export, and a "lora" run would need
its adapters merged first.

    python3 -m pdspeech_ssl.export_wav2vec2 path/to/epoch123-bal_acc0.771.ckpt out_dir/
    python3 -m pdspeech_ssl.export_wav2vec2 last.ckpt out_dir/ --base /path/to/wav2vec2-xlsr-53-espeak-cv-ft
"""

import argparse
import shutil
from pathlib import Path

import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from pdspeech_ssl.config import EncoderHParams

PREFIX = "model.wav2vec2."
# tokenizer files of the CTC base model, copied as-is so loaders that try a Wav2Vec2Processor still work
TOKENIZER_FILES = ("vocab.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json")


def export(ckpt_path: Path, out_dir: Path, base: str) -> None:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    w2v_state = {k[len(PREFIX):]: v for k, v in state.items() if k.startswith(PREFIX)}
    if not w2v_state:
        raise ValueError(f"{ckpt_path} holds no {PREFIX}* weights: was this a trainable_mode=frozen run?")
    if any("lora_" in k for k in w2v_state):
        raise ValueError(f"{ckpt_path} is a LoRA run: merge the adapters before exporting")

    model = Wav2Vec2Model.from_pretrained(base, local_files_only=True)
    missing, unexpected = model.load_state_dict(w2v_state, strict=False)
    if unexpected:
        raise ValueError(f"checkpoint weights the base model doesn't have (wrong --base?): {unexpected[:5]}")
    frozen = sorted({k.split(".")[0] for k in missing})
    print(f"loaded {len(w2v_state)} fine-tuned tensors; {len(missing)} kept from the base model ({', '.join(frozen) or 'none'})")

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    Wav2Vec2FeatureExtractor.from_pretrained(base, local_files_only=True).save_pretrained(out_dir)
    for name in TOKENIZER_FILES:
        if (Path(base) / name).exists():
            shutil.copy(Path(base) / name, out_dir / name)
    print(f"wrote {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=Path, help="Lightning .ckpt from main_ssl.py")
    parser.add_argument("out_dir", type=Path, help="HuggingFace folder to write")
    parser.add_argument("--base", default=EncoderHParams().checkpoint, help="base checkpoint the run started from")
    args = parser.parse_args()
    export(args.ckpt, args.out_dir, args.base)


if __name__ == "__main__":
    main()
