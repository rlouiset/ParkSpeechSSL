from dataclasses import dataclass, field
from typing import List, Optional

# note: intentionally `str`, not `typing.Literal` -- omegaconf's structured
# configs don't support Literal type annotations. Allowed values are
# "frozen" / "lora" / "full", validated at runtime in model.py.


@dataclass
class EncoderHParams:
    # HuggingFace checkpoint. XLSR-53 chosen for multilingual coverage
    # (our datasets span Italian, Spanish, Czech, Mandarin, English).
    checkpoint: str = "facebook/wav2vec2-large-xlsr-53"
    # "frozen": no gradients into wav2vec2 at all.
    # "lora": base weights frozen, LoRA adapters on attention q/v projections.
    # "full": every wav2vec2 weight above the CNN feature extractor is trainable
    #   (the CNN feature extractor is always frozen, standard practice for wav2vec2 fine-tuning).
    trainable_mode: str = "lora"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class ModelHParams:
    d_proj: Optional[int] = None  # projection dim before the temporal transformer (null = skip)
    blstm_layers: int = 2  # kept this name to mirror HDSNetModelHParams; feeds num_layers of the temporal transformer
    d_blstm: int = 128  # kept this name to mirror HDSNetModelHParams; feeds d_model of the temporal transformer
    d_emb: int = 32
    dropout: float = 0.2
    n_att: int = 1  # number of attention heads
    proj_head_dim: int = 32  # SimCLR-style projection head output dim (contrastive loss only, not the linear probe)


@dataclass
class AugmentHParams:
    # Everything here perturbs *recording condition*, not voice production
    # (no pitch-shift / time-stretch / formant warping -- those would destroy
    # the F0/jitter/shimmer/speech-rate biomarkers this model should learn to be
    # sensitive to for the HC/PD probe).
    noise_prob: float = 0.5
    noise_snr_db_range: List[float] = field(default_factory=lambda: [5.0, 30.0])
    gain_prob: float = 0.5
    gain_db_range: List[float] = field(default_factory=lambda: [-6.0, 6.0])
    crop_prob: float = 0.7
    crop_ratio_range: List[float] = field(default_factory=lambda: [0.8, 1.0])
    reverb_prob: float = 0.3
    reverb_decay_range: List[float] = field(default_factory=lambda: [0.1, 0.4])
    bandlimit_prob: float = 0.3
    bandlimit_sr_choices: List[int] = field(default_factory=lambda: [8000, 11025])
    # Feature-level time masking applied to the wav2vec2 output sequence
    # (same spirit as wav2vec2's own pretraining masking), independent per view.
    feature_mask_prob: float = 0.3
    feature_mask_fraction: float = 0.15


@dataclass
class DataHParams:
    derivatives_root: str = "/lustre/fsstor/projects/rech/haj/uik24xv/ParkSSLSpeechData"
    sample_rate: int = 16000
    # fraction of HC/PD individuals assigned to val; MSA/PSP/DYS individuals always go to train
    val_fraction: float = 0.25
    split_seed: int = 42
    max_audio_seconds: float = 20.0  # safety cap, matches the preprocessing pipeline's own bound
    num_workers: int = 8


@dataclass
class LossHParams:
    temperature: float = 0.1
    gather_across_gpus: bool = True


@dataclass
class TrainingHParams:
    batch_size_per_gpu: int = 64  # number of INDIVIDUALS per GPU per step (=> 2x that many views)
    lr: float = 1e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 500
    max_epochs: int = 200
    limit_train_batches: float = 1.0  # fraction (0-1) or absolute count of batches; useful for smoke tests
    limit_val_batches: float = 1.0
    precision: str = "bf16-mixed"
    devices: int = 4
    accelerator: str = "gpu"
    strategy: str = "ddp"
    accumulate_grad_batches: int = 1
    gradient_clip_val: float = 1.0
    # how often (in epochs) to run the HC/PD linear probe during validation
    probe_every_n_epochs: int = 1
    probe_lr: float = 1e-2
    probe_epochs: int = 200
    probe_weight_decay: float = 1e-4
    # random subsample caps so probe cost doesn't grow unbounded with dataset size
    # (probe runs on rank 0 only, over individual *segments*, not paired individuals)
    probe_max_train_samples: int = 4000
    probe_max_val_samples: int = 1000


@dataclass
class WandbHParams:
    project: str = "pdspeech_ssl"
    name: str = "wav2vec2xlsr_lora_transformer"
    mode: str = "online"  # overridden to "offline" via WANDB_MODE env on clusters w/o internet


@dataclass
class HParams:
    encoder: EncoderHParams = field(default_factory=EncoderHParams)
    model: ModelHParams = field(default_factory=ModelHParams)
    augment: AugmentHParams = field(default_factory=AugmentHParams)
    data: DataHParams = field(default_factory=DataHParams)
    loss: LossHParams = field(default_factory=LossHParams)
    training: TrainingHParams = field(default_factory=TrainingHParams)
    wandb: WandbHParams = field(default_factory=WandbHParams)
    seed: int = 42
