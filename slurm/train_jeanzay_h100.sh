
#!/bin/bash
#SBATCH --job-name pdspeech_ssl
#SBATCH --time=00-19:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --constraint h100
#SBATCH --account haj@h100
#SBATCH --output pdspeech_ssl_%j.txt

# NOTE: double-check --account/--constraint against your actual Jean Zay H100
# allocation name (this mirrors the @a100 example's convention, but I can't
# verify the exact current H100 account suffix/partition naming from here).

module purge
conda deactivate
module load miniforge/24.9.0
conda activate py39  # must have: torch, torchaudio, lightning, wandb, transformers==4.53.3,
                     # peft, soundfile, scikit-learn, hydra-core, omegaconf installed

export WANDB_MODE=offline
export HF_HUB_OFFLINE=1  # set to 0 for the very first run so wav2vec2-xlsr-53 can download;
                          # once cached in $HF_HOME, flip back to 1 for offline compute nodes

cd "${SLURM_SUBMIT_DIR}"

# --ntasks-per-node=4 + srun launches 4 processes, one per H100; Lightning's
# SLURM environment plugin auto-detects rank/world-size from the SLURM env
# vars, so training.devices=4 with strategy=ddp in the config just works.
srun python3 -m pdspeech_ssl.main_ssl
