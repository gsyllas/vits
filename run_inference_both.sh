#!/bin/bash
# Run inference on latest checkpoint of both runs.

set -e
cd "$(dirname "$0")"

# --- Baseline GT (only need to do this once) ---
echo "==> Extracting ground truth samples..."
python3 infer_speakers.py \
    --config configs/greek_ms_from_scratch.json \
    --gt_dir speaker_samples/baseline \
    --output_dir speaker_samples/1gpu

# --- 4-GPU run ---
echo ""
echo "==> Running 4-GPU inference on latest checkpoint"
python3 infer_speakers.py \
    --config configs/greek_ms_from_scratch_4gpu.json \
    --output_dir speaker_samples/4gpu

echo ""
echo "All done. Results:"
echo "  speaker_samples/baseline/  - ground truth wavs (5 samples x speakers 0 and 1)"
echo "  speaker_samples/1gpu/      - 1-GPU synthesis matched to baseline texts"
echo "  speaker_samples/4gpu/      - 4-GPU synthesis matched to baseline texts"
