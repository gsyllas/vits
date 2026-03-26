"""
Merge two single-speaker Greek TTS dataset folders into VITS filelists.

Each folder has: metadata.csv + wavs/ directory.
CSV format: filename,speaker_id,transcription,transcription_original,origin_dataset,gender

Output: pipe-separated filelists for VITS multi-speaker training.
Format: wav_path|speaker_int_id|text
"""

import argparse
import csv
import os
import random


def read_metadata(folder_path, speaker_int_id):
    """Read metadata.csv and return list of (wav_path, speaker_int_id, text) tuples."""
    csv_path = os.path.join(folder_path, "metadata.csv")
    entries = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wav_path = os.path.join(folder_path, row["filename"])
            text = row["transcription"].strip()
            if not text:
                continue
            entries.append((wav_path, str(speaker_int_id), text))
    return entries


def main():
    parser = argparse.ArgumentParser(description="Merge multi-speaker CSV datasets into VITS filelists")
    parser.add_argument("--datasets", nargs="+", required=True,
                        help="Paths to dataset folders (each with metadata.csv + wavs/)")
    parser.add_argument("--output_dir", default="filelists",
                        help="Directory to write output filelists")
    parser.add_argument("--prefix", default="greek_ms",
                        help="Prefix for output filenames")
    parser.add_argument("--val_ratio", type=float, default=0.005,
                        help="Fraction of data for validation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    all_entries = []
    for sid, folder in enumerate(args.datasets):
        entries = read_metadata(folder, sid)
        print(f"Speaker {sid}: {folder} — {len(entries)} utterances")
        all_entries.extend(entries)

    print(f"\nTotal utterances: {len(all_entries)}")
    n_speakers = len(args.datasets)

    # Shuffle and split
    random.shuffle(all_entries)
    n_val = max(1, int(len(all_entries) * args.val_ratio))
    val_entries = all_entries[:n_val]
    train_entries = all_entries[n_val:]

    # Write filelists
    train_path = os.path.join(args.output_dir, f"{args.prefix}_audio_sid_text_train_filelist.txt")
    val_path = os.path.join(args.output_dir, f"{args.prefix}_audio_sid_text_val_filelist.txt")

    for path, data in [(train_path, train_entries), (val_path, val_entries)]:
        with open(path, "w", encoding="utf-8") as f:
            for wav, sid, text in data:
                f.write(f"{wav}|{sid}|{text}\n")
        print(f"Written {len(data)} entries to {path}")

    print(f"\n--- Summary ---")
    print(f"n_speakers: {n_speakers}")
    print(f"Train: {len(train_entries)}, Val: {len(val_entries)}")
    print(f"\nNext steps:")
    print(f"  1. Run: python preprocess.py --filelists {train_path} {val_path} --text_index 2 --text_cleaners greek_cleaners")
    print(f"  2. Set n_speakers={n_speakers} in your config JSON")


if __name__ == "__main__":
    main()
