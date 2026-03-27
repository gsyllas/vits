# Preprocessing Debug Notes (2026-03-26)

## Problem
`run_preprocess_2spk.slurm` segfaults in espeak-ng during phonemization.

## Root Cause
The text column in `greek_ms_2spk_audio_sid_text_train_filelist.txt` contains literal `|` (pipe) characters — the same character used as the filelist delimiter.

Example broken line:
```
.../spk_0001__004458.wav|0|Σύνολο | δεκατέσσερα | διακόσια πέντε |
```

This causes `load_filepaths_and_text` (which splits on `|`) to parse 6+ fields instead of 3. `text_index=2` then grabs a wrong fragment, and espeak-ng crashes.

## Affected Lines
At least 10+ lines from the `gsyllas_dataset_normalized` speaker (sid=0) have pipes in the text. These appear to be table/list data (schedules, counts, etc.) where the normalizer converted numbers to words but kept the pipe separators.

## Fix
The `|` characters need to be removed from the source text **before** `merge_datasets.py` creates the filelists. This should be handled in the text-preprocessing normalizer's `strip_punctuation` step — either the pipe wasn't included in characters to strip, or these lines bypassed normalization.

After fixing the normalizer, re-run the full pipeline:
1. Re-normalize the source text (text-preprocessing)
2. Re-run `merge_datasets.py`
3. Re-run `sbatch run_preprocess_2spk.slurm`
