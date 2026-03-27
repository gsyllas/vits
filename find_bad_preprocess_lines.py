import argparse
import subprocess
import sys
import unicodedata

ALLOWED_PUNCTUATION = set(",.;!·…- ")

SUBPROCESS_CODE = r"""
import argparse
import sys
import text
from utils import load_filepaths_and_text


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--filelist", required=True)
  parser.add_argument("--text_index", type=int, default=2)
  parser.add_argument("--text_cleaners", nargs="+", required=True)
  parser.add_argument("--start", type=int, required=True)
  parser.add_argument("--end", type=int, required=True)
  args = parser.parse_args()

  rows = load_filepaths_and_text(args.filelist)
  for i in range(args.start, min(args.end, len(rows))):
    text._clean_text(rows[i][args.text_index], args.text_cleaners)


if __name__ == "__main__":
  main()
"""


def describe_suspicious_chars(value):
  pieces = []
  for ch in value:
    category = unicodedata.category(ch)
    codepoint = ord(ch)
    if ch in "\n\r\t":
      continue
    try:
      name = unicodedata.name(ch)
    except ValueError:
      name = "UNKNOWN"

    is_greek_letter = category.startswith("L") and "GREEK" in name
    is_ascii_digit = "0" <= ch <= "9"
    is_ascii_letter = ("A" <= ch <= "Z") or ("a" <= ch <= "z")
    is_allowed_punct = ch in ALLOWED_PUNCTUATION
    is_space = ch.isspace()

    if category.startswith("C") or ch == "|" or is_ascii_digit or is_ascii_letter:
      pieces.append(f"U+{codepoint:04X} {repr(ch)} {name} [{category}]")
      continue

    if is_greek_letter or is_allowed_punct or is_space:
      continue

    if category.startswith("L") or category.startswith("M") or category.startswith("P") or category.startswith("S"):
      try:
        pieces.append(f"U+{codepoint:04X} {repr(ch)} {name} [{category}]")
      except ValueError:
        pieces.append(f"U+{codepoint:04X} {repr(ch)} UNKNOWN [{category}]")
  return pieces


def run_chunk(filelist, text_index, cleaners, start, end):
  cmd = [
    sys.executable,
    "-c",
    SUBPROCESS_CODE,
    "--filelist",
    filelist,
    "--text_index",
    str(text_index),
    "--start",
    str(start),
    "--end",
    str(end),
    "--text_cleaners",
    *cleaners,
  ]
  return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def main():
  parser = argparse.ArgumentParser(
    description="Find filelist rows that crash preprocess cleaners by isolating them in subprocesses."
  )
  parser.add_argument("--filelist", required=True)
  parser.add_argument("--text_index", type=int, default=2)
  parser.add_argument("--text_cleaners", nargs="+", required=True)
  parser.add_argument("--chunk_size", type=int, default=200)
  parser.add_argument("--max_bad_lines", type=int, default=20)
  args = parser.parse_args()

  with open(args.filelist, encoding="utf-8") as f:
    raw_lines = [line.rstrip("\n") for line in f]

  rows = [line.split("|") for line in raw_lines]
  print(f"Loaded {len(rows)} rows from {args.filelist}")

  bad_field_rows = []
  for idx, parts in enumerate(rows, start=1):
    if len(parts) != args.text_index + 1:
      bad_field_rows.append((idx, len(parts), raw_lines[idx - 1]))

  if bad_field_rows:
    print("Rows with invalid field counts:")
    for idx, count, raw in bad_field_rows[: args.max_bad_lines]:
      print(f"  line {idx}: fields={count}")
      print(f"    {raw}")
    print("Fix field-count issues first.")
    sys.exit(1)

  bad_chunks = []
  for start in range(0, len(rows), args.chunk_size):
    end = min(start + args.chunk_size, len(rows))
    result = run_chunk(args.filelist, args.text_index, args.text_cleaners, start, end)
    if result.returncode != 0:
      bad_chunks.append((start, end, result))
      print(f"Chunk {start + 1}-{end} failed with code {result.returncode}")
      if result.stderr.strip():
        print(result.stderr.strip()[:2000])

  if not bad_chunks:
    print("No crashing chunks found.")
    return

  found = 0
  for start, end, _ in bad_chunks:
    for i in range(start, end):
      result = run_chunk(args.filelist, args.text_index, args.text_cleaners, i, i + 1)
      if result.returncode == 0:
        continue
      found += 1
      text_value = rows[i][args.text_index]
      print()
      print(f"Bad line {i + 1}:")
      print(raw_lines[i])
      suspicious = describe_suspicious_chars(text_value)
      if suspicious:
        print("  Suspicious chars:")
        for item in suspicious[:50]:
          print(f"    {item}")
      if result.stderr.strip():
        print("  stderr:")
        print(result.stderr.strip()[:4000])
      if found >= args.max_bad_lines:
        print()
        print(f"Stopping after {found} bad lines.")
        return

  if found == 0:
    print("Chunks failed, but no single crashing line was isolated.")


if __name__ == "__main__":
  main()
