"""
Convert a Hugging Face dataset saved on disk to VITS filelist format.

Expected dataset columns: audio, text, speaker_id
Output filelist format: wav_path|speaker_id_int|text

This script:
  1. Loads the dataset from disk
  2. Optionally selects a speaker subset
  3. Exports WAV files to a target directory
  4. Creates train/val filelists with speaker labels mapped to integers
  5. Reports the observed input sampling rates and exports audio at target_sr

Typical 2-speaker usage:
  python prepare_greek_data.py \
    --dataset_path /path/to/multi_speaker_combined \
    --split train \
    --val_split eval \
    --include_speakers female male \
    --output_wav_dir /path/to/greek_ms_2spk_22kmono \
    --filelist_prefix greek_ms_2spk
"""

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict

import soundfile as sf
from datasets import Dataset, DatasetDict, load_from_disk


_CONTROL_RE = re.compile(r"[\u0000-\u001f\u007f-\u009f]+")
_WHITESPACE_RE = re.compile(r"\s+")
_PROGRESS_EVERY = 1000


def clean_filelist_text(text, delimiter="|"):
    """Keep raw punctuation, only remove filelist-unsafe/control characters."""
    if text is None:
        return ""
    text = str(text).replace(delimiter, " ")
    text = _CONTROL_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def load_split(dataset_obj, split_name):
    if isinstance(dataset_obj, DatasetDict):
        if split_name not in dataset_obj:
            available = ", ".join(sorted(dataset_obj.keys()))
            raise ValueError(f"Split '{split_name}' not found. Available splits: {available}")
        return dataset_obj[split_name]

    if split_name not in (None, "", "train"):
        raise ValueError(
            f"--split/--val_split was set to '{split_name}', but the saved dataset is a single Dataset, not a DatasetDict."
        )
    if not isinstance(dataset_obj, Dataset):
        raise TypeError(f"Unsupported dataset type: {type(dataset_obj)!r}")
    return dataset_obj


def find_speaker_mapping_path(dataset_path, explicit_path=None):
    if explicit_path:
        return explicit_path
    candidate = os.path.join(dataset_path, "speaker_id_mapping.json")
    if os.path.exists(candidate):
        return candidate
    return None


def load_source_speaker_mapping(mapping_path):
    if not mapping_path:
        return {}, {}

    with open(mapping_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    raw_mapping = payload.get("mapping") or payload.get("speaker_to_id") or {}
    label_to_raw = {str(label): str(raw_value) for label, raw_value in raw_mapping.items()}
    raw_to_label = {raw_value: label for label, raw_value in label_to_raw.items()}
    return label_to_raw, raw_to_label


def resolve_requested_speakers(include_speakers, label_to_raw, raw_to_label):
    if not include_speakers:
        return None

    resolved = []
    for requested in include_speakers:
        requested = str(requested)
        raw_value = label_to_raw.get(requested, requested)
        resolved.append(str(raw_value))

    pretty = [f"{requested}->{raw_to_label.get(raw, raw)}({raw})"
              for requested, raw in zip(include_speakers, resolved)]
    print("Resolved requested speakers:", ", ".join(pretty), flush=True)
    return resolved


def speaker_label(raw_speaker, raw_to_label):
    raw_speaker = str(raw_speaker)
    return raw_to_label.get(raw_speaker, raw_speaker)


def build_speaker_set(speaker_durations, requested_raw_speakers, min_speaker_hours):
    if requested_raw_speakers:
        speakers = []
        for spk in requested_raw_speakers:
            hours = speaker_durations.get(spk, 0.0) / 3600.0
            if spk not in speaker_durations:
                print(f"  Requested speaker not found: {spk}")
                continue
            if hours < min_speaker_hours:
                print(f"  Skipping speaker {spk}: {hours:.2f}h < {min_speaker_hours}h")
                continue
            speakers.append(spk)
        return speakers

    speakers = []
    for spk, dur in sorted(speaker_durations.items()):
        hours = dur / 3600.0
        if hours >= min_speaker_hours:
            speakers.append(spk)
        else:
            print(f"  Skipping speaker {spk}: {hours:.2f}h < {min_speaker_hours}h")
    return speakers


def scan_dataset(ds, args, speaker_durations, sample_rate_counts, requested_raw_speakers):
    requested_set = set(requested_raw_speakers) if requested_raw_speakers else None
    matched = 0
    for i, example in enumerate(ds, start=1):
        spk = str(example[args.speaker_column])
        if requested_set and spk not in requested_set:
            continue
        audio = example[args.audio_column]
        sr = int(audio["sampling_rate"])
        duration = len(audio["array"]) / sr
        speaker_durations[spk] += duration
        sample_rate_counts[sr] += 1
        matched += 1
        if matched % _PROGRESS_EVERY == 0:
            print(
                f"  scan progress: visited={i}, matched={matched}, last_sr={sr}",
                flush=True,
            )


def export_entries(ds, split_name, spk_to_id, args):
    split_wav_dir = os.path.join(args.output_wav_dir, split_name)
    os.makedirs(split_wav_dir, exist_ok=True)

    librosa = None
    entries = []
    resampled = 0

    for i, example in enumerate(ds):
        spk = str(example[args.speaker_column])
        if spk not in spk_to_id:
            continue

        text = clean_filelist_text(example[args.text_column])
        if not text:
            continue

        audio = example[args.audio_column]
        array = audio["array"]
        sr = int(audio["sampling_rate"])

        if sr != args.target_sr:
            if librosa is None:
                import librosa
            array = librosa.resample(array, orig_sr=sr, target_sr=args.target_sr)
            resampled += 1

        spk_int = spk_to_id[spk]
        wav_filename = f"spk{spk_int:04d}_{i:07d}.wav"
        wav_path = os.path.join(split_wav_dir, wav_filename)
        sf.write(wav_path, array, args.target_sr)
        entries.append((wav_path, str(spk_int), text))

        if (i + 1) % _PROGRESS_EVERY == 0:
            print(f"  [{split_name}] processed {i + 1} examples ...", flush=True)

    return entries, resampled


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
                        help="Dataset split to use when loading training data from a DatasetDict")
    parser.add_argument("--val_split", default=None,
                        help="Optional validation split to use directly from a DatasetDict")
    parser.add_argument("--include_speakers", nargs="+", default=None,
                        help="Optional speaker labels to keep, in the exact order they should be mapped")
    parser.add_argument("--speaker_mapping_json", default=None,
                        help="Optional JSON file that maps human-readable speaker labels to dataset speaker_id values")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_wav_dir, exist_ok=True)
    os.makedirs(args.output_filelist_dir, exist_ok=True)

    print(f"Loading dataset from {args.dataset_path} ...", flush=True)
    ds_obj = load_from_disk(args.dataset_path)

    train_ds = load_split(ds_obj, args.split)
    val_ds = load_split(ds_obj, args.val_split) if args.val_split else None

    source_mapping_path = None
    label_to_raw, raw_to_label = {}, {}
    if args.speaker_column == "speaker_id":
        source_mapping_path = find_speaker_mapping_path(args.dataset_path, args.speaker_mapping_json)
        label_to_raw, raw_to_label = load_source_speaker_mapping(source_mapping_path)
        if source_mapping_path:
            print(f"Loaded source speaker mapping: {source_mapping_path}", flush=True)
    else:
        print(f"Filtering speakers directly with column: {args.speaker_column}", flush=True)
        if args.speaker_mapping_json:
            print(
                f"Ignoring --speaker_mapping_json because --speaker_column={args.speaker_column} "
                f"already contains speaker labels.",
                flush=True,
            )

    requested_raw_speakers = resolve_requested_speakers(args.include_speakers, label_to_raw, raw_to_label)

    speaker_durations = defaultdict(float)  # seconds
    sample_rate_counts = Counter()
    print("Scanning dataset for speaker durations and sampling rates ...", flush=True)
    scan_dataset(train_ds, args, speaker_durations, sample_rate_counts, requested_raw_speakers)
    if val_ds is not None:
        scan_dataset(val_ds, args, speaker_durations, sample_rate_counts, requested_raw_speakers)

    valid_speakers = build_speaker_set(
        speaker_durations=speaker_durations,
        requested_raw_speakers=requested_raw_speakers,
        min_speaker_hours=args.min_speaker_hours,
    )
    if not valid_speakers:
        raise ValueError("No speakers matched the requested filters.")

    # Create integer speaker ID mapping
    spk_to_id = {spk: idx for idx, spk in enumerate(valid_speakers)}
    speaker_name_to_id = {speaker_label(spk, raw_to_label): idx for spk, idx in spk_to_id.items()}
    n_speakers = len(valid_speakers)
    print(f"Total speakers after filtering: {n_speakers}", flush=True)
    for spk in valid_speakers:
        print(
            f"  speaker {speaker_label(spk, raw_to_label)!r} "
            f"(source id {spk}) -> {spk_to_id[spk]} "
            f"({speaker_durations[spk] / 3600.0:.2f}h)",
            flush=True,
        )

    if sample_rate_counts:
        sr_summary = ", ".join(
            f"{sr}Hz={count}" for sr, count in sorted(sample_rate_counts.items())
        )
        print(f"Observed input sampling rates: {sr_summary}", flush=True)

    # Save speaker mapping
    mapping_path = os.path.join(args.output_filelist_dir, f"{args.filelist_prefix}_speaker_map.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "speaker_to_id": speaker_name_to_id,
                "mapping": speaker_name_to_id,
                "source_speaker_values": {speaker_label(spk, raw_to_label): spk for spk in valid_speakers},
                "n_speakers": n_speakers,
                "num_speakers": n_speakers,
                "source_dataset": args.dataset_path,
                "train_split": args.split,
                "val_split": args.val_split,
                "target_sr": args.target_sr,
                "source_speaker_mapping_json": source_mapping_path,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Speaker mapping saved to {mapping_path}", flush=True)

    # Export WAVs and build filelist entries.
    if val_ds is not None:
        print(f"Exporting training split '{args.split}' ...", flush=True)
        train_entries, train_resampled = export_entries(train_ds, args.split, spk_to_id, args)
        print(f"Exporting validation split '{args.val_split}' ...", flush=True)
        val_entries, val_resampled = export_entries(val_ds, args.val_split, spk_to_id, args)
        random.shuffle(train_entries)
        random.shuffle(val_entries)
    else:
        print(f"Exporting source split '{args.split}' ...", flush=True)
        entries, total_resampled = export_entries(train_ds, args.split, spk_to_id, args)
        random.shuffle(entries)
        n_val = max(1, int(len(entries) * args.val_ratio))
        val_entries = entries[:n_val]
        train_entries = entries[n_val:]
        train_resampled = total_resampled
        val_resampled = 0

    print(f"Total valid train entries: {len(train_entries)}", flush=True)
    print(f"Total valid val entries: {len(val_entries)}", flush=True)

    # Write filelists
    train_path = os.path.join(args.output_filelist_dir,
                              f"{args.filelist_prefix}_audio_sid_text_train_filelist.txt")
    val_path = os.path.join(args.output_filelist_dir,
                            f"{args.filelist_prefix}_audio_sid_text_val_filelist.txt")

    for path, data in [(train_path, train_entries), (val_path, val_entries)]:
        with open(path, "w", encoding="utf-8") as f:
            for wav, sid, text in data:
                f.write(f"{wav}|{sid}|{text}\n")
        print(f"Written {len(data)} entries to {path}", flush=True)

    print(f"\n--- Summary ---", flush=True)
    print(f"n_speakers: {n_speakers}", flush=True)
    print(f"Train samples: {len(train_entries)}", flush=True)
    print(f"Val samples: {len(val_entries)}", flush=True)
    if any(sr != args.target_sr for sr in sample_rate_counts):
        print(
            f"Input audio was not uniformly {args.target_sr} Hz; exported WAVs were resampled to {args.target_sr} Hz "
            f"(train={train_resampled}, val={val_resampled}).",
            flush=True,
        )
    else:
        print(f"All included input audio was already {args.target_sr} Hz.", flush=True)
    print(
        "Text was kept mostly raw; only control characters, repeated whitespace, and the filelist delimiter '|' were normalized.",
        flush=True,
    )
    print(f"Set n_speakers={n_speakers} in your config JSON.", flush=True)


if __name__ == "__main__":
    main()
