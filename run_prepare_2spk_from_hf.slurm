#!/bin/bash
#SBATCH --job-name=vits-prepare-2spk-hf
#SBATCH --account=EUHPC_D29_081
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH --time=08:00:00
#SBATCH --output=/leonardo_work/EUHPC_D29_081/gsyllas0/vits/logs/%x_%j.out
#SBATCH --error=/leonardo_work/EUHPC_D29_081/gsyllas0/vits/logs/%x_%j.err

set -euo pipefail

module purge
module load cuda/12.2
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vits_orig

VITS_DIR="/leonardo_work/EUHPC_D29_081/gsyllas0/vits"
HF_DATASET="/leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/multi_speaker_combined"
OUTPUT_WAV_DIR="/leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/greek_ms_2spk_22kmono"

cd "${VITS_DIR}"
mkdir -p logs

echo "$(date): Preparing female/male subset from ${HF_DATASET}"

python -u prepare_greek_data.py \
  --dataset_path "${HF_DATASET}" \
  --split train \
  --val_split eval \
  --speaker_column speaker_name \
  --include_speakers female male \
  --output_wav_dir "${OUTPUT_WAV_DIR}" \
  --output_filelist_dir filelists \
  --filelist_prefix greek_ms_2spk \
  --target_sr 22050

echo "$(date): Dataset preparation done"
echo "Next step: sbatch run_preprocess_2spk_safe.slurm"
