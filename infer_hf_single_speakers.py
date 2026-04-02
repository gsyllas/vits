"""
Run inference for the single-speaker Hugging Face VITS/MMS fine-tuned models.

This script is meant for the Greek female and male Leonardo runs trained with
the separate `finetune-hf-vits` repository. It synthesizes the same set of
sentences for both models so their voices can be compared directly.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from scipy.io.wavfile import write
from transformers import AutoConfig, AutoFeatureExtractor, AutoTokenizer, VitsModel, pipeline


DEFAULT_FEMALE_MODEL_DIR = (
    "/leonardo_work/EUHPC_D29_081/gsyllas0/output/"
    "single_speaker_female_mms_finetuning"
)
DEFAULT_MALE_MODEL_DIR = (
    "/leonardo_work/EUHPC_D29_081/gsyllas0/output/"
    "single_speaker_male_mms_finetuning"
)
DEFAULT_SENTENCES = [
    "Καλημέρα, πώς είσαι σήμερα;",
    "Σήμερα δοκιμάζουμε το μοντέλο σύνθεσης φωνής στα ελληνικά.",
    "Η ποιότητα της ομιλίας πρέπει να είναι καθαρή και φυσική.",
    "Θέλω να συγκρίνω τη γυναικεία και την ανδρική φωνή με το ίδιο κείμενο.",
    "Αυτή είναι μια σύντομη δοκιμή για την τελική αξιολόγηση του συστήματος.",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--female_model_dir", default=DEFAULT_FEMALE_MODEL_DIR)
    parser.add_argument("--male_model_dir", default=DEFAULT_MALE_MODEL_DIR)
    parser.add_argument("--output_dir", default="speaker_samples/hf_single_speakers")
    parser.add_argument(
        "--text",
        action="append",
        default=None,
        help="Sentence to synthesize. Pass multiple times for multiple sentences.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU id for inference. Use -1 for CPU.",
    )
    return parser.parse_args()


def find_latest_checkpoint(model_dir):
    checkpoints = []
    for path in Path(model_dir).glob("checkpoint-*"):
        if path.is_dir():
            try:
                step = int(path.name.split("-", 1)[1])
            except (IndexError, ValueError):
                continue
            checkpoints.append((step, path))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda item: item[0])
    return checkpoints[-1][1]


def has_final_model_bundle(model_dir):
    model_dir = Path(model_dir)
    has_config = (model_dir / "config.json").is_file()
    has_weights = any((model_dir / name).is_file() for name in ["model.safetensors", "pytorch_model.bin"])
    return has_config and has_weights


def resolve_model_source(model_dir):
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    if has_final_model_bundle(model_dir):
        return {
            "bundle_dir": model_dir,
            "weights_dir": model_dir,
            "source": "final_model",
        }

    checkpoint_dir = find_latest_checkpoint(model_dir)
    if checkpoint_dir is None:
        raise FileNotFoundError(
            f"No final model bundle or checkpoint-* directories found in {model_dir}"
        )

    if not any((checkpoint_dir / name).is_file() for name in ["model.safetensors", "pytorch_model.bin"]):
        raise FileNotFoundError(f"No model weights found in latest checkpoint: {checkpoint_dir}")

    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"Missing config.json in model root: {model_dir}")

    return {
        "bundle_dir": model_dir,
        "weights_dir": checkpoint_dir,
        "source": f"checkpoint:{checkpoint_dir.name}",
    }


def build_synthesizer(model_dir, device):
    resolved = resolve_model_source(model_dir)

    if resolved["weights_dir"] == resolved["bundle_dir"]:
        synthesizer = pipeline(
            "text-to-speech",
            model=str(resolved["bundle_dir"]),
            device=device,
        )
    else:
        config = AutoConfig.from_pretrained(str(resolved["bundle_dir"]))
        tokenizer = AutoTokenizer.from_pretrained(str(resolved["bundle_dir"]))
        feature_extractor = AutoFeatureExtractor.from_pretrained(str(resolved["bundle_dir"]))
        model = VitsModel.from_pretrained(str(resolved["weights_dir"]), config=config)
        if device >= 0 and torch.cuda.is_available():
            model = model.to(f"cuda:{device}")

        synthesizer = pipeline(
            "text-to-speech",
            model=model,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            device=device,
        )

    return synthesizer, resolved


def ensure_int16(audio):
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    if hasattr(audio, "dtype") and str(audio.dtype) == "int16":
        return audio
    if hasattr(audio, "dtype") and str(audio.dtype).startswith("float"):
        clipped = audio.clip(-1.0, 1.0)
        return (clipped * 32767.0).astype("int16")
    return audio


def write_manifest(output_dir, sentences):
    manifest_path = Path(output_dir) / "sentences.txt"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for i, sentence in enumerate(sentences, start=1):
            f.write(f"sent{i:02d}|{sentence}\n")


def write_model_info(output_dir, model_infos):
    info_path = Path(output_dir) / "model_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(model_infos, f, ensure_ascii=False, indent=2)


def synthesize_model(name, model_dir, output_root, sentences, device):
    model_output_dir = Path(output_root) / name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    synthesizer, resolved = build_synthesizer(model_dir, device)
    print(f"\n--- {name} ---")
    print(f"Model dir: {model_dir}")
    print(f"Using: {resolved['weights_dir']} ({resolved['source']})")

    for i, sentence in enumerate(sentences, start=1):
        speech = synthesizer(sentence)
        audio = speech["audio"]
        if getattr(audio, "ndim", 1) > 1:
            audio = audio[0]
        audio = ensure_int16(audio)

        out_path = model_output_dir / f"sent{i:02d}.wav"
        write(out_path, rate=speech["sampling_rate"], data=audio)
        print(f"  sent{i:02d} -> {out_path}")

    write_manifest(model_output_dir, sentences)
    return {
        "model_dir": str(model_dir),
        "bundle_dir": str(resolved["bundle_dir"]),
        "weights_dir": str(resolved["weights_dir"]),
        "source": resolved["source"],
    }


def main():
    args = parse_args()
    sentences = args.text if args.text else DEFAULT_SENTENCES
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device >= 0 and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but no GPU is available.")

    model_infos = {}
    model_infos["female"] = synthesize_model(
        "female",
        args.female_model_dir,
        output_dir,
        sentences,
        args.device,
    )
    model_infos["male"] = synthesize_model(
        "male",
        args.male_model_dir,
        output_dir,
        sentences,
        args.device,
    )

    write_manifest(output_dir, sentences)
    write_model_info(output_dir, model_infos)
    print("\nDone.")


if __name__ == "__main__":
    main()
