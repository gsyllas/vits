"""
Portable VITS multispeaker inference from a local manifest file.

Use this when you have already copied:
  - the VITS repo
  - a config JSON
  - a checkpoint directory or file
  - a manifest of sample texts (for example speaker_samples/baseline/sentences.txt)

The manifest format is one entry per line:
    sent01|text
or:
    spk00_sent01|text
"""

import argparse
import glob
import os

import torch
from scipy.io.wavfile import write

import commons
import utils
from models import SynthesizerTrn
from text import cleaned_text_to_sequence
from text.cleaners import greek_cleaners
from text.symbols import symbols


DEFAULT_SPEAKERS = [0, 1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="If omitted, use the latest G_*.pth from model_dir")
    parser.add_argument("--model_dir", default=None,
                        help="Directory that contains G_*.pth checkpoints")
    parser.add_argument("--output_dir", default="speaker_samples/portable_vits")
    parser.add_argument("--speakers", type=int, nargs="+", default=DEFAULT_SPEAKERS)
    parser.add_argument("--device", default="cuda",
                        help="Torch device, e.g. cuda, cuda:0, cpu")
    parser.add_argument("--noise_scale", type=float, default=0.667)
    parser.add_argument("--noise_scale_w", type=float, default=0.8)
    parser.add_argument("--length_scale", type=float, default=1.0)
    parser.add_argument(
        "--manifest_text_mode",
        choices=["auto", "cleaned", "raw"],
        default="auto",
        help="Interpret manifest text as already-cleaned phonemes, raw Greek text, or auto-detect.",
    )
    parser.add_argument(
        "--keep_existing_outputs",
        action="store_true",
        help="Do not clear old spk*_sent*.wav files from the output dir before writing.",
    )
    return parser.parse_args()


SYMBOL_SET = set(symbols)


def is_cleaned_text(text):
    return all(ch in SYMBOL_SET for ch in text)


def normalize_text(text, text_mode):
    if text_mode == "cleaned":
        return text
    if text_mode == "raw":
        return greek_cleaners(text)
    return text if is_cleaned_text(text) else greek_cleaners(text)


def get_text(text, hps, text_mode):
    cleaned = normalize_text(text, text_mode)
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
    return utils.latest_checkpoint_path(model_dir, "G_*.pth")


def parse_manifest_label(label, index):
    label = label.strip() or f"sent{index:02d}"
    if label.startswith("spk") and "_" in label:
        speaker_prefix, item_label = label.split("_", 1)
        speaker_id = speaker_prefix[3:]
        if speaker_id.isdigit():
            return int(speaker_id), item_label
    return None, label


def load_manifest(path):
    shared_items = []
    speaker_items = {}
    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("|", 1)
            if len(parts) == 2:
                label, text = parts
            else:
                label, text = f"sent{idx:02d}", parts[0]
            speaker_id, item_label = parse_manifest_label(label, idx)
            item = {
                "label": item_label,
                "text": text.strip(),
            }
            if speaker_id is None:
                shared_items.append(item)
            else:
                speaker_items.setdefault(speaker_id, []).append(item)
    if not shared_items and not speaker_items:
        raise ValueError(f"No manifest entries found in {path}")
    return shared_items, speaker_items


def get_items_for_speaker(speaker_id, shared_items, speaker_items):
    if speaker_id in speaker_items:
        return speaker_items[speaker_id]
    if shared_items:
        return shared_items
    raise ValueError(f"No manifest entries found for speaker {speaker_id}")


def cleanup_generated_files(output_dir):
    for pattern in ["spk*_sent*.wav"]:
        for path in glob.glob(os.path.join(output_dir, pattern)):
            if os.path.isfile(path):
                os.remove(path)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if not args.keep_existing_outputs:
        cleanup_generated_files(args.output_dir)

    hps = utils.get_hparams_from_file(args.config)
    device = torch.device(args.device)
    checkpoint_path = resolve_checkpoint(
        args.checkpoint,
        resolve_model_dir(args.config, args.model_dir),
    )
    shared_items, speaker_items = load_manifest(args.manifest)

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).to(device)
    net_g.eval()
    utils.load_checkpoint(checkpoint_path, net_g, None)

    with torch.no_grad():
        for sid in args.speakers:
            sid_tensor = torch.LongTensor([sid]).to(device)
            items = get_items_for_speaker(sid, shared_items, speaker_items)
            for item in items:
                stn_tst = get_text(item["text"], hps, args.manifest_text_mode)
                x_tst = stn_tst.to(device).unsqueeze(0)
                x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)

                audio = net_g.infer(
                    x_tst,
                    x_tst_lengths,
                    sid=sid_tensor,
                    noise_scale=args.noise_scale,
                    noise_scale_w=args.noise_scale_w,
                    length_scale=args.length_scale,
                )[0][0, 0].data.cpu().float().numpy()

                out_path = os.path.join(args.output_dir, f"spk{sid:02d}_{item['label']}.wav")
                write(out_path, hps.data.sampling_rate, audio)
                print(f"spk{sid:02d} {item['label']} -> {out_path}")


if __name__ == "__main__":
    main()
