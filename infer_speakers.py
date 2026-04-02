"""
Speaker-identity check: synthesizes baseline validation texts for selected
speakers and optionally copies the matching ground-truth wavs for 1:1
comparison.

Usage:
    python3 infer_speakers.py \
        --config configs/greek_ms_from_scratch_4gpu.json \
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

DEFAULT_SPEAKERS = [0, 1]
DEFAULT_SAMPLES_PER_SPEAKER = 5


def get_text(text, hps):
    cleaned = greek_cleaners(text)
    text_norm = cleaned_text_to_sequence(cleaned)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    return torch.LongTensor(text_norm)


def resolve_model_dir(config_path, model_dir=None):
    if model_dir:
        return model_dir
    if os.path.basename(config_path) == "config.json":
        return os.path.dirname(os.path.abspath(config_path))
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    return os.path.join("logs", config_name)


def resolve_checkpoint(checkpoint_path, model_dir):
    if checkpoint_path:
        return checkpoint_path
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    try:
        return utils.latest_checkpoint_path(model_dir, "G_*.pth")
    except IndexError as exc:
        raise FileNotFoundError(
            f"No generator checkpoints found in {model_dir}"
        ) from exc


def get_source_filelists(hps, filelist_override=None):
    if filelist_override:
        return [filelist_override]

    filelists = []
    for candidate in [hps.data.validation_files, hps.data.training_files]:
        if candidate and candidate not in filelists:
            filelists.append(candidate)
    return filelists


def collect_speaker_examples(filelist_paths, speaker_ids, samples_per_speaker):
    examples = {sid: [] for sid in speaker_ids}
    seen_wavs = set()

    for filelist_path in filelist_paths:
        with open(filelist_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|", 2)
                if len(parts) < 3:
                    continue
                wav_path, sid_str, text = parts
                try:
                    sid = int(sid_str)
                except ValueError:
                    continue
                if sid not in examples or len(examples[sid]) >= samples_per_speaker:
                    continue
                if wav_path in seen_wavs or not os.path.exists(wav_path):
                    continue

                examples[sid].append({
                    "wav_path": wav_path,
                    "text": text,
                })
                seen_wavs.add(wav_path)

                if all(len(examples[s]) >= samples_per_speaker for s in speaker_ids):
                    break
        if all(len(examples[s]) >= samples_per_speaker for s in speaker_ids):
            break

    missing = [
        f"speaker {sid}: found {len(samples)}/{samples_per_speaker}"
        for sid, samples in examples.items()
        if len(samples) < samples_per_speaker
    ]
    if missing:
        details = ", ".join(missing)
        raise RuntimeError(
            "Could not collect enough baseline samples from "
            f"{', '.join(filelist_paths)}: {details}"
        )
    return examples


def write_manifest(output_dir, examples):
    manifest_path = os.path.join(output_dir, "sentences.txt")
    with open(manifest_path, "w", encoding="utf-8") as f:
        for sid, samples in examples.items():
            for i, sample in enumerate(samples, start=1):
                f.write(f"spk{sid:02d}_sent{i:02d}|{sample['text']}\n")


def copy_gt_samples(examples, gt_dir):
    os.makedirs(gt_dir, exist_ok=True)
    for sid, samples in examples.items():
        for i, sample in enumerate(samples, start=1):
            dst = os.path.join(gt_dir, f"spk{sid:02d}_sent{i:02d}.wav")
            shutil.copy(sample["wav_path"], dst)
            print(f"  GT spk{sid:02d} sent{i:02d} -> {dst}")
    write_manifest(gt_dir, examples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="If omitted, use the latest G_*.pth from model_dir")
    parser.add_argument("--model_dir", default=None,
                        help="Override the model directory used to resolve the latest checkpoint")
    parser.add_argument("--output_dir", default="speaker_samples/run")
    parser.add_argument("--gt_dir", default=None,
                        help="If set, copy matching GT wavs here from the source filelist")
    parser.add_argument("--filelist", default=None,
                        help="If omitted, scan validation first and then training from the config")
    parser.add_argument("--speakers", type=int, nargs="+", default=DEFAULT_SPEAKERS)
    parser.add_argument("--samples_per_speaker", type=int, default=DEFAULT_SAMPLES_PER_SPEAKER)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    hps = utils.get_hparams_from_file(args.config)
    speaker_ids = list(dict.fromkeys(args.speakers))
    filelist_paths = get_source_filelists(hps, args.filelist)
    model_dir = resolve_model_dir(args.config, args.model_dir)
    checkpoint_path = resolve_checkpoint(args.checkpoint, model_dir)
    examples = collect_speaker_examples(filelist_paths, speaker_ids, args.samples_per_speaker)

    print(f"\n--- Using examples from: {', '.join(filelist_paths)} ---")
    print(f"Speakers: {speaker_ids}")
    print(f"Samples per speaker: {args.samples_per_speaker}")

    # Ground truth baseline
    if args.gt_dir:
        print(f"\n--- Extracting GT samples -> {args.gt_dir} ---")
        copy_gt_samples(examples, args.gt_dir)

    # Load model
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).cuda()
    net_g.eval()
    utils.load_checkpoint(checkpoint_path, net_g, None)

    print(
        f"\n--- Synthesizing {len(speaker_ids)} speakers x "
        f"{args.samples_per_speaker} baseline sentences ---"
    )
    print(f"Checkpoint: {checkpoint_path}")

    with torch.no_grad():
        for sid in speaker_ids:
            sid_tensor = torch.LongTensor([sid]).cuda()
            for i, sample in enumerate(examples[sid], start=1):
                stn_tst = get_text(sample["text"], hps)
                x_tst = stn_tst.cuda().unsqueeze(0)
                x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()

                audio = net_g.infer(
                    x_tst, x_tst_lengths, sid=sid_tensor,
                    noise_scale=0.667, noise_scale_w=0.8, length_scale=1
                )[0][0, 0].data.cpu().float().numpy()

                out_path = os.path.join(args.output_dir, f"spk{sid:02d}_sent{i:02d}.wav")
                write(out_path, hps.data.sampling_rate, audio)
                print(f"  spk{sid:02d} sent{i:02d} -> {out_path}")

    write_manifest(args.output_dir, examples)

    print("\nDone.")


if __name__ == "__main__":
    main()
