#!/bin/bash
#SBATCH --job-name=vits-greek-ms-4gpu
#SBATCH --account=EUHPC_D29_081
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=/leonardo_work/EUHPC_D29_081/gsyllas0/vits/logs/%x_%j.out
#SBATCH --error=/leonardo_work/EUHPC_D29_081/gsyllas0/vits/logs/%x_%j.err
#SBATCH --signal=B:USR1@600

# ============================================================
# VITS from-scratch — 4x A100 64GB DDP — Greek multi-speaker
# Combined datasets: CommonVoice + CS10 + male3h
#
# train_ms.py uses mp.spawn() with NCCL backend.
# Auto-detects 4 GPUs, batch_size=32 per GPU (128 effective).
# Auto-resubmits before 24h timeout for continuous training.
# Checkpoints saved every 1000 steps, auto-resume on restart.
# ============================================================

# --- Timeout handler: save & requeue ---
handle_timeout() {
    echo "$(date): Caught USR1 — 10 min before timeout. Killing training gracefully..."
    kill -TERM "${TRAIN_PID}" 2>/dev/null
    wait "${TRAIN_PID}" 2>/dev/null
    echo "$(date): Training stopped. Resubmitting job..."
    sbatch "$0"
}
trap 'handle_timeout' USR1

# --- Modules & Environment ---
module purge
module load cuda/12.2
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vits_orig

export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
# NCCL tuning for Leonardo
export TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-1048576}
export NCCL_DEBUG=WARN

# --- Paths ---
VITS_DIR="/leonardo_work/EUHPC_D29_081/gsyllas0/vits"
CONFIG="${VITS_DIR}/configs/greek_ms_from_scratch_4gpu.json"
MODEL_NAME="greek_ms_from_scratch_4gpu"

cd "${VITS_DIR}"
mkdir -p logs

echo "============================================"
echo "Job: ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "Config: ${CONFIG}"
echo "Model dir: logs/${MODEL_NAME}"
echo "============================================"

# Check if resuming
if ls "logs/${MODEL_NAME}"/G_*.pth 1>/dev/null 2>&1; then
    echo "Found existing checkpoints — will resume training"
else
    echo "No checkpoints found — starting from scratch"
fi

# --- Launch training in background so trap can fire ---
python train_ms.py \
    -c "${CONFIG}" \
    -m "${MODEL_NAME}" &

TRAIN_PID=$!
wait "${TRAIN_PID}"
EXIT_CODE=$?

echo "$(date): Training finished with exit code ${EXIT_CODE}"
