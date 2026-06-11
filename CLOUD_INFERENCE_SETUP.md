# Cloud Inference Runbook

This note is the minimal setup for moving the models off Leonardo and running
inference on a new GPU provider.

It assumes:

- you only want the latest checkpoints, not full training history
- for the old VITS models you only care about speakers `0` and `1`
- for the old VITS models you want to reuse the downloaded baseline
  `sentences.txt` manifest

## What To Download From Leonardo

### Old VITS checkpoints

You only need the latest generator checkpoint from each model:

- latest `G_*.pth` from `/leonardo_work/EUHPC_D29_081/gsyllas0/vits/logs/greek_ms_from_scratch`
- latest `G_*.pth` from `/leonardo_work/EUHPC_D29_081/gsyllas0/vits/logs/greek_ms_from_scratch_4gpu`

You do not need:

- older `G_*.pth` checkpoints
- any `D_*.pth` discriminator checkpoints

### Old VITS baseline manifest

Download at least:

- `/leonardo_work/EUHPC_D29_081/gsyllas0/vits/speaker_samples/baseline/sentences.txt`

Optional, if you also want GT wavs for listening comparison:

- `/leonardo_work/EUHPC_D29_081/gsyllas0/vits/speaker_samples/baseline`

### HF fine-tuned single-speaker models

For each of these two directories:

- `/leonardo_work/EUHPC_D29_081/gsyllas0/output/single_speaker_female_mms_finetuning`
- `/leonardo_work/EUHPC_D29_081/gsyllas0/output/single_speaker_male_mms_finetuning`

download:

- the root files such as `config.json`, tokenizer files, and final weights if
  they exist
- only the newest `checkpoint-*` directory if there is no final root model
  bundle

You do not need all older `checkpoint-*` folders.

## Local Download Commands

Run these on your local machine, not on Leonardo.

First create the local folders:

```bash
mkdir -p ~/greek_tts/vits/logs/greek_ms_from_scratch
mkdir -p ~/greek_tts/vits/logs/greek_ms_from_scratch_4gpu
mkdir -p ~/greek_tts/vits/speaker_samples/baseline
mkdir -p ~/greek_tts/output/single_speaker_female_mms_finetuning
mkdir -p ~/greek_tts/output/single_speaker_male_mms_finetuning
```

Set a base path and resolve the latest old-VITS checkpoints:

```bash
BASE=/leonardo_work/EUHPC_D29_081/gsyllas0

LATEST_1GPU=$(ssh gsyllas0@login.leonardo.cineca.it "cd $BASE/vits/logs/greek_ms_from_scratch && ls -1v G_*.pth | tail -n 1")
LATEST_4GPU=$(ssh gsyllas0@login.leonardo.cineca.it "cd $BASE/vits/logs/greek_ms_from_scratch_4gpu && ls -1v G_*.pth | tail -n 1")
```

Download the latest old-VITS checkpoints and the baseline manifest:

```bash
scp "gsyllas0@data.leonardo.cineca.it:$BASE/vits/logs/greek_ms_from_scratch/$LATEST_1GPU" \
  ~/greek_tts/vits/logs/greek_ms_from_scratch/

scp "gsyllas0@data.leonardo.cineca.it:$BASE/vits/logs/greek_ms_from_scratch_4gpu/$LATEST_4GPU" \
  ~/greek_tts/vits/logs/greek_ms_from_scratch_4gpu/

scp "gsyllas0@data.leonardo.cineca.it:$BASE/vits/speaker_samples/baseline/sentences.txt" \
  ~/greek_tts/vits/speaker_samples/baseline/
```

If you also want the GT wavs:

```bash
scp -r gsyllas0@data.leonardo.cineca.it:$BASE/vits/speaker_samples/baseline \
  ~/greek_tts/vits/speaker_samples/
```

Download the HF model roots without old checkpoint history:

```bash
rsync -av --exclude 'checkpoint-*' \
  gsyllas0@data.leonardo.cineca.it:$BASE/output/single_speaker_female_mms_finetuning/ \
  ~/greek_tts/output/single_speaker_female_mms_finetuning/

rsync -av --exclude 'checkpoint-*' \
  gsyllas0@data.leonardo.cineca.it:$BASE/output/single_speaker_male_mms_finetuning/ \
  ~/greek_tts/output/single_speaker_male_mms_finetuning/
```

If the root folder does not contain final weights, resolve and download only the
newest checkpoint too:

```bash
LATEST_FEMALE=$(ssh gsyllas0@login.leonardo.cineca.it "cd $BASE/output/single_speaker_female_mms_finetuning && ls -1d checkpoint-* | sort -V | tail -n 1")
LATEST_MALE=$(ssh gsyllas0@login.leonardo.cineca.it "cd $BASE/output/single_speaker_male_mms_finetuning && ls -1d checkpoint-* | sort -V | tail -n 1")

scp -r "gsyllas0@data.leonardo.cineca.it:$BASE/output/single_speaker_female_mms_finetuning/$LATEST_FEMALE" \
  ~/greek_tts/output/single_speaker_female_mms_finetuning/

scp -r "gsyllas0@data.leonardo.cineca.it:$BASE/output/single_speaker_male_mms_finetuning/$LATEST_MALE" \
  ~/greek_tts/output/single_speaker_male_mms_finetuning/
```

## What To Upload To The New Provider

Upload this layout:

```text
~/greek_tts/
  vits/
    ...this repo...
    logs/
      greek_ms_from_scratch/
        G_xxxxx.pth
      greek_ms_from_scratch_4gpu/
        G_xxxxx.pth
    speaker_samples/
      baseline/
        sentences.txt
        optional GT wavs...
  output/
    single_speaker_female_mms_finetuning/
    single_speaker_male_mms_finetuning/
```

## Environment For Old VITS Inference

```bash
sudo apt-get update
sudo apt-get install -y espeak-ng build-essential

cd ~/greek_tts/vits
python3 -m venv .venv-vits
source .venv-vits/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.vits_inference.txt
cd monotonic_align
python setup.py build_ext
find build -name '*.so' -exec cp {} . \;
cd ..
python -m py_compile infer_speakers.py infer_speakers_from_manifest.py
```

## Environment For HF Single-Speaker Inference

```bash
cd ~/greek_tts/vits
python3 -m venv .venv-hf
source .venv-hf/bin/activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.hf_inference.txt
python -m py_compile infer_hf_single_speakers.py
```

## Inference Commands We Will Run

### Old VITS 4-GPU model

```bash
cd ~/greek_tts/vits
source .venv-vits/bin/activate
python infer_speakers_from_manifest.py \
  --config configs/greek_ms_from_scratch_4gpu.json \
  --checkpoint logs/greek_ms_from_scratch_4gpu/G_xxxxx.pth \
  --manifest speaker_samples/baseline/sentences.txt \
  --manifest_text_mode cleaned \
  --speakers 0 1 \
  --output_dir speaker_samples/cloud_4gpu
```

### Old VITS 1-GPU model

```bash
cd ~/greek_tts/vits
source .venv-vits/bin/activate
python infer_speakers_from_manifest.py \
  --config configs/greek_ms_from_scratch.json \
  --checkpoint logs/greek_ms_from_scratch/G_xxxxx.pth \
  --manifest speaker_samples/baseline/sentences.txt \
  --manifest_text_mode cleaned \
  --speakers 0 1 \
  --output_dir speaker_samples/cloud_1gpu
```

Replace `G_xxxxx.pth` with the exact downloaded filename.

### HF female and male models

```bash
cd ~/greek_tts/vits
source .venv-hf/bin/activate
python infer_hf_single_speakers.py \
  --female_model_dir ~/greek_tts/output/single_speaker_female_mms_finetuning \
  --male_model_dir ~/greek_tts/output/single_speaker_male_mms_finetuning \
  --output_dir ~/greek_tts/output/hf_single_speaker_samples
```

## Output Folders

After inference, the expected outputs are:

- `~/greek_tts/vits/speaker_samples/cloud_4gpu`
- `~/greek_tts/vits/speaker_samples/cloud_1gpu`
- `~/greek_tts/output/hf_single_speaker_samples/female`
- `~/greek_tts/output/hf_single_speaker_samples/male`

## Notes

- The old VITS path is not fully standalone. It needs this repo because it uses
  local modules such as `models`, `utils`, `text`, and `monotonic_align`.
- The HF helper is much lighter, but still easiest to run from this repo where
  the script already lives.
- For the old VITS models, the manifest-driven script is the cleanest portable
  option because it does not require copying the original training filelists and
  dataset wavs.
