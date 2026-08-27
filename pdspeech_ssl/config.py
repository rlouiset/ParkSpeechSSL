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
    trainable_mode: str = "full"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class ModelHParams:
    d_proj: Optional[int] = None  # projection dim before the BiLSTM temporal encoder (null = skip)
    blstm_layers: int = 2  # num_layers of the BiLSTM temporal encoder
    d_blstm: int = 128  # hidden size per direction of the BiLSTM temporal encoder
    d_emb: int = 32
    dropout: float = 0.2
    proj_head_dim: int = 32  # SimCLR-style projection head output dim (contrastive loss only, not the linear probe)


@dataclass
class AugmentHParams:
    noise_prob: float = 0.25
    noise_snr_db_range: List[float] = field(
        default_factory=lambda: [18.0, 30.0]
    )

    gain_prob: float = 0.25
    gain_db_range: List[float] = field(
        default_factory=lambda: [-3.0, 3.0]
    )

    crop_prob: float = 0.0
    crop_ratio_range: List[float] = field(
        default_factory=lambda: [0.95, 1.0]
    )

    reverb_prob: float = 0.10
    reverb_decay_range: List[float] = field(
        default_factory=lambda: [0.10, 0.20]
    )

    bandlimit_prob: float = 0.10
    bandlimit_sr_choices: List[int] = field(
        default_factory=lambda: [11025]
    )

    feature_mask_prob: float = 0.10
    feature_mask_fraction: float = 0.05


@dataclass
class DataHParams:
    derivatives_root: str = "/lustre/fswork/projects/rech/haj/uik24xv/ParkSSLSpeechData"
    sample_rate: int = 16000
    # fraction of HC/PD individuals assigned to val; MSA/PSP/DYS individuals always go to train
    val_fraction: float = 0.2
    split_seed: int = 42
    max_audio_seconds: float = 20.0  # hard cap: waveforms are truncated to this length in load_waveform
    # LUFS integrated-loudness normalization target (EBU R128 broadcast default); a single
    # global per-clip gain, applied in load_waveform before augmentation. Removes absolute
    # level (heavily confounded by this project's per-corpus recording setups) while leaving
    # within-clip dynamic range/loudness variability untouched (gain-invariant by construction).
    # Set to None to disable.
    target_lufs: Optional[float] = -23.0
    num_workers: int = 8


@dataclass
class LossHParams:
    temperature: float = 0.1
    gather_across_gpus: bool = True


@dataclass
class DiseaseHParams:
    # Only used when training.objective == "disease_uniformity". Splits the SimCLR-style
    # proj_head output (dim = ModelHParams.proj_head_dim) into Z_D = proj[:d_disease]
    # (disease-specific -- forced to 0 for HC individuals) and Z_C = proj[d_disease:]
    # (shared/common -- age, sex, smoking, recording condition, etc, unconstrained).
    # Must be < proj_head_dim.
    d_disease: int = 8
    leaky_slope: float = 0.2  # applied to the Z_D block only, before the final renormalize --
    # keeps disease deviation mostly one-sided so HC (pinned at Z_D=0) can't end up
    # geometrically "between" two disease subgroups spread into opposing directions.
    uniformity_t: float = 2.0  # Wang & Isola's default
    align_weight: float = 1.0
    uniform_weight: float = 1.0


@dataclass
class TrainingHParams:
    # "simclr": NT-Xent contrastive loss aligning the two augmented views of the same
    #   individual (IndividualPairDataset's view1[i]/view2[i]) against all other
    #   individuals in the (globally-gathered) batch as negatives. HC/PD is evaluated
    #   only via the rank-0 linear probe below, never backpropagated into the encoder.
    # "hc_vs_rest_bce": direct supervised HC-vs-rest (PD/MSA/PSP/DYS) binary classification
    #   loss on both views, backpropagated straight into the encoder -- a reachability
    #   sanity check, kept available via configs/hc_vs_rest.yaml to relaunch on demand.
    # "disease_uniformity": alignment+uniformity (Wang & Isola) on a normalized proj_head
    #   output split into Z_D/Z_C (see DiseaseHParams) -- HC individuals' Z_D is forced to
    #   0 (pinned to the Z_C-only equatorial subsphere), letting non-HC individuals' Z_D
    #   norm emerge as an unsupervised severity signal. Kept available via
    #   configs/disease_uniformity.yaml.
    # validated at runtime in lightning_module.py.
    objective: str = "simclr"
    batch_size_per_gpu: int = 16  # number of INDIVIDUALS per GPU per step (=> 2x that many views)
    lr: float = 3e-4  # bumped 3x from 1e-4 -- worth watching for instability now that trainable_mode="frozen"
    weight_decay: float = 1e-2
    # ~370 train individuals / 4 GPUs / batch_size_per_gpu=16 => only ~5-6 optimizer
    # steps/epoch; 500 would take ~100 epochs just to finish warmup, so scale it down
    # to match this dataset's step-count regime (~20 epochs of warmup instead).
    warmup_steps: int = 100
    max_epochs: int = 200
    limit_train_batches: float = 1.0  # fraction (0-1) or absolute count of batches; useful for smoke tests
    limit_val_batches: float = 1.0
    precision: str = "bf16-mixed"
    devices: int = 4
    accelerator: str = "gpu"
    strategy: str = "ddp"
    # after max_audio_seconds dropped to 10s (~4x less attention memory on top of the
    # batch_size_per_gpu 64->16 cut), there's headroom again -- accumulate_grad_batches=4
    # on top of only ~5-6 micro-batches/epoch/GPU was leaving just ~1 real optimizer step
    # per epoch, which is why the loss/probe were flat (still deep in warmup after 48 epochs).
    accumulate_grad_batches: int = 1
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
    name: str = "wav2vec2xlsr_full_blstm_simclr"
    mode: str = "online"  # overridden to "offline" via WANDB_MODE env on clusters w/o internet


@dataclass
class HParams:
    encoder: EncoderHParams = field(default_factory=EncoderHParams)
    model: ModelHParams = field(default_factory=ModelHParams)
    augment: AugmentHParams = field(default_factory=AugmentHParams)
    data: DataHParams = field(default_factory=DataHParams)
    loss: LossHParams = field(default_factory=LossHParams)
    disease: DiseaseHParams = field(default_factory=DiseaseHParams)
    training: TrainingHParams = field(default_factory=TrainingHParams)
    wandb: WandbHParams = field(default_factory=WandbHParams)
    seed: int = 42
