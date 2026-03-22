"""
Convert HuggingFace dataset (multi_speaker_combined) to VITS filelist format.

Expected HF dataset columns: audio, text, speaker_id
Output filelist format: wav_path|speaker_id_int|text

This script:
  1. Loads the HF dataset from disk
  2. Exports WAV files to a target directory
  3. Creates train/val filelists with speaker_id mapped to integers
  4. Optionally filters by min_speaker_hours
"""

import argparse
import os
import json
import random
from collections import defaultdict

import soundfile as sf
from datasets import load_from_disk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True,
                        help="Path to HF dataset saved on disk")
    parser.add_argument("--output_wav_dir", required=True,
                        help="Directory to export WAV files (22050 Hz)")
    parser.add_argument("--output_filelist_dir", default="filelists",
                        help="Directory to write filelists")
    parser.add_argument("--filelist_prefix", default="greek_ms",
                        help="Prefix for filelist filenames")
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--audio_column", default="audio")
    parser.add_argument("--speaker_column", default="speaker_id")
    parser.add_argument("--val_ratio", type=float, default=0.005,
                        help="Fraction of data to use for validation")
    parser.add_argument("--min_speaker_hours", type=float, default=0.0,
                        help="Minimum hours per speaker to include")
    parser.add_argument("--target_sr", type=int, default=22050)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="train",
                        help="Dataset split to use")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_wav_dir, exist_ok=True)
    os.makedirs(args.output_filelist_dir, exist_ok=True)

    print(f"Loading dataset from {args.dataset_path} ...")
    ds = load_from_disk(args.dataset_path)
    if isinstance(ds, dict):
        ds = ds[args.split]

    # Build speaker mapping and compute durations
    speaker_durations = defaultdict(float)  # seconds
    print("Scanning dataset for speaker durations ...")
    for i, example in enumerate(ds):
        audio = example[args.audio_column]
        sr = audio["sampling_rate"]
        duration = len(audio["array"]) / sr
        spk = str(example[args.speaker_column])
        speaker_durations[spk] += duration

    # Filter speakers by min hours
    valid_speakers = set()
    for spk, dur in speaker_durations.items():
        hours = dur / 3600.0
        if hours >= args.min_speaker_hours:
            valid_speakers.add(spk)
        else:
            print(f"  Skipping speaker {spk}: {hours:.2f}h < {args.min_speaker_hours}h")

    # Create integer speaker ID mapping
    sorted_speakers = sorted(valid_speakers)
    spk_to_id = {spk: idx for idx, spk in enumerate(sorted_speakers)}
    n_speakers = len(sorted_speakers)
    print(f"Total speakers after filtering: {n_speakers}")

    # Save speaker mapping
    mapping_path = os.path.join(args.output_filelist_dir, f"{args.filelist_prefix}_speaker_map.json")
    with open(mapping_path, "w") as f:
        json.dump({"speaker_to_id": spk_to_id, "n_speakers": n_speakers}, f, indent=2)
    print(f"Speaker mapping saved to {mapping_path}")

    # Export WAVs and build filelist entries
    import librosa
    entries = []
    for i, example in enumerate(ds):
        spk = str(example[args.speaker_column])
        if spk not in valid_speakers:
            continue

        audio = example[args.audio_column]
        array = audio["array"]
        sr = audio["sampling_rate"]

        # Resample if needed
        if sr != args.target_sr:
            array = librosa.resample(array, orig_sr=sr, target_sr=args.target_sr)

        text = example[args.text_column].strip()
        if not text:
            continue

        spk_int = spk_to_id[spk]
        wav_filename = f"spk{spk_int:04d}_{i:07d}.wav"
        wav_path = os.path.join(args.output_wav_dir, wav_filename)

        sf.write(wav_path, array, args.target_sr)
        entries.append((wav_path, str(spk_int), text))

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1} examples ...")

    print(f"Total valid entries: {len(entries)}")

    # Shuffle and split
    random.shuffle(entries)
    n_val = max(1, int(len(entries) * args.val_ratio))
    val_entries = entries[:n_val]
    train_entries = entries[n_val:]

    # Write filelists
    train_path = os.path.join(args.output_filelist_dir,
                              f"{args.filelist_prefix}_audio_sid_text_train_filelist.txt")
    val_path = os.path.join(args.output_filelist_dir,
                            f"{args.filelist_prefix}_audio_sid_text_val_filelist.txt")

    for path, data in [(train_path, train_entries), (val_path, val_entries)]:
        with open(path, "w", encoding="utf-8") as f:
            for wav, sid, text in data:
                f.write(f"{wav}|{sid}|{text}\n")
        print(f"Written {len(data)} entries to {path}")

    print(f"\n--- Summary ---")
    print(f"n_speakers: {n_speakers}")
    print(f"Train samples: {len(train_entries)}")
    print(f"Val samples: {len(val_entries)}")
    print(f"Set n_speakers={n_speakers} in your config JSON.")


if __name__ == "__main__":
    main()
