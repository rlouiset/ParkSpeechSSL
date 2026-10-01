from dataclasses import dataclass, field
from typing import List, Optional

# note: intentionally `str`, not `typing.Literal`, for every enum-like field below --
# omegaconf's structured configs don't support Literal type annotations. Allowed
# values are listed next to each field and validated at runtime.


@dataclass
class EncoderHParams:
    # Local copy of facebook/wav2vec2-large-xlsr-53 (loaded with local_files_only=True).
    # XLSR-53 chosen for multilingual coverage (our datasets span Italian, Spanish,
    # Czech, Mandarin, English). Must be the *pretraining* checkpoint (quantizer,
    # project_q, project_hid present) -- asserted at load time in model.py.
    model_name_or_path: str = "/lustre/fswork/projects/rech/haj/uik24xv/huggingface/wav2vec2-large-xlsr-53"
    # "frozen": no gradients into wav2vec2 at all, backbone kept in eval mode.
    # "full": Transformer, feature_projection, masked_spec_embed and project_hid are
    #   trainable. The CNN feature encoder is always frozen (standard practice for
    #   wav2vec2 fine-tuning); quantizer/project_q follow W2V2HParams.train_quantizer.
    # validated at runtime in model.py.
    trainable_mode: str = "full"
    # Recompute Transformer-layer activations in backward instead of storing them:
    # same results (dropout RNG is replayed), much less memory, ~30% slower step.
    # Needed for w2v2.simclr_view=unmasked (4 backbone graphs alive per step).
    gradient_checkpointing: bool = False


@dataclass
class ModelHParams:
    d_proj: Optional[int] = None  # projection dim before the BiLSTM temporal encoder (null = skip)
    blstm_layers: int = 2  # num_layers of the BiLSTM temporal encoder
    d_blstm: int = 128  # hidden size per direction of the BiLSTM temporal encoder
    d_emb: int = 32
    dropout: float = 0.2
    proj_head_dim: int = 32  # SimCLR-style projection head output dim (contrastive loss only, not the linear probe)
    # BiLSTM input: "last" = final wav2vec2 hidden state; "weighted" = softmax-weighted
    # sum of all hidden states (CNN-projection output + every Transformer layer), with
    # learnable weights initialized uniform. validated at runtime in model.py.
    layer_aggregation: str = "weighted"


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
    # "cross_segment" (default): each individual's contrastive pair is two *distinct*
    #   segments/recordings when >=2 exist (random.sample), falling back to the same
    #   segment augmented twice only for individuals with a single segment (e.g. every
    #   FredPrior patient). Pushes the encoder toward invariance across different
    #   utterances/recordings of the same subject, on top of augmentation invariance.
    # "within_segment": classic SimCLR instance discrimination -- always the *same*
    #   single segment for both views (two independent augment_waveform draws), even
    #   for individuals with multiple segments. Isolates pure augmentation-invariance
    #   (recording condition: noise/gain/reverb/bandwidth) and drops the cross-segment
    #   invariance pressure entirely -- useful as an ablation against "cross_segment".
    # validated at runtime in data.py (IndividualPairDataset.PAIR_MODES).
    pair_mode: str = "cross_segment"


@dataclass
class LossHParams:
    temperature: float = 0.1
    gather_across_gpus: bool = True
    # weight of the training.objective loss (simclr / hc_vs_rest_bce) in the total loss:
    # total = simclr_weight * objective_loss + w2v2.weight * w2v2_loss
    simclr_weight: float = 1.0
    # Auxiliary supervised HC-vs-rest (PD/MSA/PSP/DYS) hinge loss on a linear cls_head over
    # embd, added on top of training.objective=simclr:
    #   + hc_vs_rest_hinge_weight * mean_over_views(max(0, margin - y * logit)), y = -1 HC / +1 rest.
    # 0 disables it (no cls_head is built).
    hc_vs_rest_hinge_weight: float = 0.0
    hc_vs_rest_hinge_margin: float = 1.0


@dataclass
class W2V2HParams:
    # wav2vec2 masked-prediction pretraining loss, trained jointly with training.objective.
    weight: float = 1.0  # 0 disables the w2v2 loss entirely (no masking, no quantizer forward)
    # span masking over CNN frames (HF _compute_mask_indices); wav2vec2 paper defaults
    mask_prob: float = 0.65
    mask_length: int = 10
    min_masks: int = 2
    num_negatives: int = 100  # distractors per masked frame, sampled within the same utterance
    diversity_loss_weight: float = 0.1
    gumbel_temperature: float = 0.5  # only used when train_quantizer=True
    # False: quantizer + project_q frozen *and* kept in eval mode, so targets are
    # deterministic (argmax codevectors, no Gumbel noise).
    train_quantizer: bool = False
    # Which forward feeds the SimCLR/objective head:
    # "masked": the same masked forward that computes the w2v2 loss (1 backbone pass per view).
    # "unmasked": a separate unmasked forward for the head, plus the masked one for the
    #   w2v2 loss only (2x backbone cost -- ablation).
    # validated at runtime in lightning_module.py.
    simclr_view: str = "masked"


@dataclass
class OptimHParams:
    lr_backbone: float = 2e-5  # wav2vec2 params (Transformer, feature_projection, masked_spec_embed, project_hid)
    lr_head: float = 1e-3  # layer weights, projector, BiLSTM, pooling, embedding_projector, proj_head, cls_head
    lr_quantizer: float = 1e-5  # quantizer + project_q, only when w2v2.train_quantizer=True
    weight_decay: float = 1e-2  # not applied to biases, LayerNorm params or the layer-weight vector
    # linear warmup then cosine decay to 0 over trainer.estimated_stepping_batches (per optimizer step)
    warmup_steps: int = 100
    # backbone LR held at 0 for the first N optimizer steps (lets the randomly initialized
    # head settle first); its warmup+cosine schedule then starts from step N.
    freeze_backbone_steps: int = 0


@dataclass
class TrainingHParams:
    # "simclr": NT-Xent contrastive loss aligning the two augmented views of the same
    #   individual (IndividualPairDataset's view1[i]/view2[i]) against all other
    #   individuals in the (globally-gathered) batch as negatives. HC/PD is evaluated
    #   only via the rank-0 linear probe below, never backpropagated into the encoder.
    # "hc_vs_rest_bce": direct supervised HC-vs-rest (PD/MSA/PSP/DYS) binary classification
    #   loss on both views, backpropagated straight into the encoder -- a reachability
    #   sanity check, kept available via configs/hc_vs_rest.yaml to relaunch on demand.
    # validated at runtime in lightning_module.py.
    objective: str = "simclr"
    batch_size_per_gpu: int = 16  # number of INDIVIDUALS per GPU per step (=> 2x that many views)
    max_epochs: int = 200
    gradient_clip_val: float = 1.0
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
    name: str = "wav2vec2xlsr_full_blstm_sepclr"
    mode: str = "online"  # overridden to "offline" via WANDB_MODE env on clusters w/o internet


@dataclass
class HParams:
    encoder: EncoderHParams = field(default_factory=EncoderHParams)
    model: ModelHParams = field(default_factory=ModelHParams)
    augment: AugmentHParams = field(default_factory=AugmentHParams)
    data: DataHParams = field(default_factory=DataHParams)
    loss: LossHParams = field(default_factory=LossHParams)
    w2v2: W2V2HParams = field(default_factory=W2V2HParams)
    optim: OptimHParams = field(default_factory=OptimHParams)
    training: TrainingHParams = field(default_factory=TrainingHParams)
    wandb: WandbHParams = field(default_factory=WandbHParams)
    seed: int = 42
