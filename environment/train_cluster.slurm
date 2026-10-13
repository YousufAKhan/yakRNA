#!/bin/bash -l
# ---------------------------------------------------------------------------
# yakRNA Design - 8 GPU single-node training (the released 110M recipe).
#
# Adapt --partition, --account and the module/environment lines to your
# cluster, then:   sbatch environment/train_cluster.slurm
# ---------------------------------------------------------------------------
#SBATCH --job-name=yakrna_train
#SBATCH --partition=gpu           # <-- your GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

CONFIG=${CONFIG:-configs/train_110M.yaml}

echo "[INFO] Job ${SLURM_JOB_ID:-local} on $(hostname), config ${CONFIG}, start $(date)"

# --- modules (cluster-specific; harmless if absent) ------------------------
if command -v module >/dev/null 2>&1; then
  module purge || true
  module load cuda/12.1 || true
fi

# --- environment -----------------------------------------------------------
# Whichever of these you used in INSTALL.md / TRAINING.md:
if command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook -s bash)"
  micromamba activate yakrna
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate yakrna
fi

# --- runtime tuning --------------------------------------------------------
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${OMP_NUM_THREADS}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Copy the dataset to node-local scratch by setting this in the config instead:
#   data_files.temporary_processing_directory: "$L_SCRATCH/yakRNA"

NPROC_PER_NODE=${SLURM_GPUS_ON_NODE:-8}
MASTER_PORT=${MASTER_PORT:-29501}

command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

srun --ntasks=1 torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  rna_gen/rna_gen_training.py \
  --config "${CONFIG}"

echo "[INFO] Finished at $(date)"
