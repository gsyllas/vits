#!/bin/bash
#SBATCH --job-name=vits-resample-2spk
#SBATCH --account=EUHPC_D29_081
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:0
#SBATCH --time=06:00:00
#SBATCH --output=/leonardo_work/EUHPC_D29_081/gsyllas0/vits/logs/%x_%j.out
#SBATCH --error=/leonardo_work/EUHPC_D29_081/gsyllas0/vits/logs/%x_%j.err

set -euo pipefail

# ============================================================
# Resample the two normalized speaker datasets to 22050 Hz mono
# for the Greek 2-speaker VITS training setup.
#
# Source datasets:
#   /leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/female_normalized
#   /leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/male_normalized
#
# Output datasets:
#   /leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/female_normalized_22kmono
#   /leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/male_normalized_22kmono
# ============================================================

DATA_DIR="/leonardo_work/EUHPC_D29_081/gsyllas0/data/tts"
LOG_DIR="/leonardo_work/EUHPC_D29_081/gsyllas0/vits/logs"
SRC_DATASETS=("female_normalized" "male_normalized")
TARGET_SR=22050
TARGET_CH=1
WORKERS="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p "${LOG_DIR}"

echo "============================================"
echo "Job: ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node: $(hostname)"
echo "Workers: ${WORKERS}"
echo "Data dir: ${DATA_DIR}"
echo "============================================"

if ! command -v ffmpeg >/dev/null 2>&1; then
  module load ffmpeg >/dev/null 2>&1 || module load FFmpeg >/dev/null 2>&1 || true
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ERROR: ffmpeg not found in PATH." >&2
  echo "Load an ffmpeg module or install ffmpeg in your environment, then retry." >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found in PATH." >&2
  exit 1
fi

echo "ffmpeg: $(command -v ffmpeg)"
ffmpeg -version | head -1
echo "python3: $(command -v python3)"
python3 --version
echo
df -h "${DATA_DIR}"
echo

for ds in "${SRC_DATASETS[@]}"; do
  src_dir="${DATA_DIR}/${ds}"
  src_wavs="${src_dir}/wavs"
  src_meta="${src_dir}/metadata.csv"
  out_dir="${DATA_DIR}/${ds}_22kmono"
  out_wavs="${out_dir}/wavs"
  out_meta="${out_dir}/metadata.csv"

  if [[ ! -d "${src_dir}" ]]; then
    echo "ERROR: missing source dataset dir: ${src_dir}" >&2
    exit 1
  fi
  if [[ ! -d "${src_wavs}" ]]; then
    echo "ERROR: missing source wav dir: ${src_wavs}" >&2
    exit 1
  fi
  if [[ ! -f "${src_meta}" ]]; then
    echo "ERROR: missing metadata.csv: ${src_meta}" >&2
    exit 1
  fi

  mkdir -p "${out_wavs}"
  cp -f "${src_meta}" "${out_meta}"

  src_count=$(find "${src_wavs}" -type f -name '*.wav' | wc -l)
  out_count=$(find "${out_wavs}" -type f -name '*.wav' | wc -l)

  echo "[${ds}]"
  echo "  source wavs: ${src_count}"
  echo "  existing output wavs: ${out_count}"
  du -sh "${src_dir}" || true
  echo
done

resample_dataset() {
  local ds="$1"
  local src_root="${DATA_DIR}/${ds}/wavs"
  local dst_root="${DATA_DIR}/${ds}_22kmono/wavs"

  echo "Starting resample for ${ds}"

  find "${src_root}" -type f -name '*.wav' -print0 | \
    xargs -0 -I{} -P "${WORKERS}" bash -c '
      set -euo pipefail
      src="$1"
      src_root="$2"
      dst_root="$3"
      target_sr="$4"
      target_ch="$5"
      rel="${src#"$src_root"/}"
      dst="${dst_root}/${rel}"
      mkdir -p "$(dirname "$dst")"
      if [[ -s "$dst" ]]; then
        exit 0
      fi
      ffmpeg -nostdin -y -loglevel error -i "$src" -ac "$target_ch" -ar "$target_sr" "$dst"
    ' _ {} "${src_root}" "${dst_root}" "${TARGET_SR}" "${TARGET_CH}"

  echo "Finished resample for ${ds}"
}

for ds in "${SRC_DATASETS[@]}"; do
  resample_dataset "${ds}"
done

echo
echo "Post-run verification"
for ds in "${SRC_DATASETS[@]}"; do
  src_wavs="${DATA_DIR}/${ds}/wavs"
  out_wavs="${DATA_DIR}/${ds}_22kmono/wavs"
  src_count=$(find "${src_wavs}" -type f -name '*.wav' | wc -l)
  out_count=$(find "${out_wavs}" -type f -name '*.wav' | wc -l)
  echo "[${ds}] source=${src_count} output=${out_count}"

  sample_wav=$(find "${out_wavs}" -type f -name '*.wav' | sort | head -1)
  if [[ -z "${sample_wav}" ]]; then
    echo "ERROR: no output wavs found for ${ds}" >&2
    exit 1
  fi

  python3 - "${sample_wav}" <<'PY'
import sys
import wave

path = sys.argv[1]
with wave.open(path, "rb") as w:
    print(f"  sample: {path}")
    print(f"  sample_rate: {w.getframerate()}")
    print(f"  channels: {w.getnchannels()}")
    print(f"  sample_width_bytes: {w.getsampwidth()}")
    print(f"  duration_sec: {round(w.getnframes() / w.getframerate(), 3)}")
PY
  echo
done

echo "$(date): Resampling complete."
