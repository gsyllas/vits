#!/bin/bash
#SBATCH --job-name=vits-preprocess-2spk
#SBATCH --account=EUHPC_D29_081
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH --time=04:00:00
#SBATCH --output=/leonardo_work/EUHPC_D29_081/gsyllas0/vits/logs/%x_%j.out
#SBATCH --error=/leonardo_work/EUHPC_D29_081/gsyllas0/vits/logs/%x_%j.err

# ============================================================
# Phonemize filelists with greek_cleaners (espeak-ng)
# ============================================================

module purge
module load cuda/12.2
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vits_orig

VITS_DIR="/leonardo_work/EUHPC_D29_081/gsyllas0/vits"
cd "${VITS_DIR}"
mkdir -p logs

echo "$(date): Starting preprocessing"

python preprocess.py \
  --filelists \
    filelists/greek_ms_2spk_audio_sid_text_train_filelist.txt \
    filelists/greek_ms_2spk_audio_sid_text_val_filelist.txt \
  --text_index 2 \
  --text_cleaners greek_cleaners

echo "$(date): Preprocessing done (exit code $?)"
