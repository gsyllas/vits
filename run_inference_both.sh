#!/bin/bash
# Run inference on latest checkpoint of both runs.
# Run from: /leonardo_work/EUHPC_D29_081/gsyllas0/vits

set -e
cd /leonardo_work/EUHPC_D29_081/gsyllas0/vits

# --- Baseline GT (only need to do this once) ---
echo "==> Extracting ground truth samples..."
CKPT_1GPU=$(ls -t logs/greek_ms_from_scratch/G_*.pth | head -1)
python infer_speakers.py \
    --config configs/greek_ms_from_scratch.json \
    --checkpoint "$CKPT_1GPU" \
    --gt_dir speaker_samples/baseline \
    --output_dir speaker_samples/1gpu

# --- 4-GPU run ---
CKPT_4GPU=$(ls -t logs/greek_ms_from_scratch_4gpu/G_*.pth | head -1)
echo ""
echo "==> 4-GPU latest checkpoint: $CKPT_4GPU"
python infer_speakers.py \
    --config configs/greek_ms_from_scratch_4gpu.json \
    --checkpoint "$CKPT_4GPU" \
    --output_dir speaker_samples/4gpu

echo ""
echo "All done. Results:"
echo "  speaker_samples/baseline/  - ground truth wavs"
echo "  speaker_samples/1gpu/      - 1-GPU synthesis (2 sentences x 6 speakers)"
echo "  speaker_samples/4gpu/      - 4-GPU synthesis (2 sentences x 6 speakers)"
