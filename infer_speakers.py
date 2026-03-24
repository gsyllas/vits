"""
Speaker-identity check: synthesizes 2 sentences per speaker and optionally
extracts a ground truth sample per speaker from the validation filelist.

Usage:
    python infer_speakers.py \
        --config configs/greek_ms_from_scratch_4gpu.json \
        --checkpoint logs/greek_ms_from_scratch_4gpu/G_84000.pth \
        --output_dir speaker_samples/4gpu \
        --gt_dir speaker_samples/baseline
"""
import argparse
import os
import shutil
import torch
from scipy.io.wavfile import write

import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import cleaned_text_to_sequence
from text.cleaners import greek_cleaners

SENTENCES = [
    "Η φωνή μου είναι μοναδική.",
    "Το σύστημα αυτό μαθαίνει να μιλά.",
]


def get_text(text, hps):
    cleaned = greek_cleaners(text)
    text_norm = cleaned_text_to_sequence(cleaned)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    return torch.LongTensor(text_norm)


def extract_gt_samples(val_filelist, gt_dir, n_speakers):
    """Copy one GT wav per speaker from the validation filelist."""
    os.makedirs(gt_dir, exist_ok=True)
    found = {}
    try:
        with open(val_filelist) as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) < 2:
                    continue
                wav_path, sid = parts[0], int(parts[1])
                if sid not in found and os.path.exists(wav_path):
                    dst = os.path.join(gt_dir, f"spk{sid:02d}_gt.wav")
                    shutil.copy(wav_path, dst)
                    found[sid] = dst
                    print(f"  GT spk{sid:02d} -> {dst}")
                if len(found) == n_speakers:
                    break
    except FileNotFoundError:
        print(f"  Warning: val filelist not found: {val_filelist}")
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="speaker_samples/run")
    parser.add_argument("--gt_dir", default=None,
                        help="If set, copy one GT wav per speaker here from val filelist")
    parser.add_argument("--speakers", type=int, nargs="+", default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    hps = utils.get_hparams_from_file(args.config)
    speaker_ids = args.speakers if args.speakers else list(range(hps.data.n_speakers))

    # Ground truth baseline
    if args.gt_dir:
        print(f"\n--- Extracting GT samples -> {args.gt_dir} ---")
        extract_gt_samples(hps.data.validation_files, args.gt_dir, hps.data.n_speakers)

    # Load model
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).cuda()
    net_g.eval()
    utils.load_checkpoint(args.checkpoint, net_g, None)

    print(f"\n--- Synthesizing {len(speaker_ids)} speakers x {len(SENTENCES)} sentences ---")
    print(f"Checkpoint: {args.checkpoint}")

    with torch.no_grad():
        for sid in speaker_ids:
            sid_tensor = torch.LongTensor([sid]).cuda()
            for i, sentence in enumerate(SENTENCES):
                stn_tst = get_text(sentence, hps)
                x_tst = stn_tst.cuda().unsqueeze(0)
                x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()

                audio = net_g.infer(
                    x_tst, x_tst_lengths, sid=sid_tensor,
                    noise_scale=0.667, noise_scale_w=0.8, length_scale=1
                )[0][0, 0].data.cpu().float().numpy()

                out_path = os.path.join(args.output_dir, f"spk{sid:02d}_sent{i+1}.wav")
                write(out_path, hps.data.sampling_rate, audio)
                print(f"  spk{sid:02d} sent{i+1} -> {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
