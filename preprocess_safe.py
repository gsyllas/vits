import argparse
import json
import subprocess
import sys

from utils import load_filepaths_and_text


WORKER_CODE = r"""
import argparse
import json
import text
from utils import load_filepaths_and_text


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--filelist", required=True)
  parser.add_argument("--text_index", type=int, required=True)
  parser.add_argument("--start", type=int, required=True)
  parser.add_argument("--end", type=int, required=True)
  parser.add_argument("--text_cleaners", nargs="+", required=True)
  args = parser.parse_args()

  rows = load_filepaths_and_text(args.filelist)
  for i in range(args.start, min(args.end, len(rows))):
    cleaned = text._clean_text(rows[i][args.text_index], args.text_cleaners)
    print(json.dumps(cleaned, ensure_ascii=False))


if __name__ == "__main__":
  main()
"""


def run_chunk(filelist, text_index, cleaners, start, end):
  cmd = [
    sys.executable,
    "-c",
    WORKER_CODE,
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


def process_range(filelist, rows, text_index, cleaners, start, end, cleaned_map, bad_rows):
  result = run_chunk(filelist, text_index, cleaners, start, end)
  if result.returncode == 0:
    outputs = [json.loads(line) for line in result.stdout.splitlines()]
    expected = end - start
    if len(outputs) != expected:
      raise RuntimeError(
        f"Worker returned {len(outputs)} lines for range {start + 1}-{end}, expected {expected}"
      )
    for offset, cleaned in enumerate(outputs):
      cleaned_map[start + offset] = cleaned
    return

  if end - start == 1:
    bad_rows.append({
      "line_number": start + 1,
      "raw": "|".join(rows[start]),
      "text": rows[start][text_index],
      "stderr": result.stderr.strip(),
      "returncode": result.returncode,
    })
    return

  mid = start + (end - start) // 2
  process_range(filelist, rows, text_index, cleaners, start, mid, cleaned_map, bad_rows)
  process_range(filelist, rows, text_index, cleaners, mid, end, cleaned_map, bad_rows)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--out_extension", default="cleaned")
  parser.add_argument("--text_index", default=1, type=int)
  parser.add_argument("--filelists", nargs="+", required=True)
  parser.add_argument("--text_cleaners", nargs="+", required=True)
  parser.add_argument("--chunk_size", default=100, type=int)
  parser.add_argument("--bad_extension", default="bad")
  args = parser.parse_args()

  for filelist in args.filelists:
    print("START:", filelist, flush=True)
    rows = load_filepaths_and_text(filelist)

    bad_field_rows = []
    for i, row in enumerate(rows, start=1):
      if len(row) != args.text_index + 1:
        bad_field_rows.append((i, len(row), "|".join(row)))

    if bad_field_rows:
      print("Invalid field counts detected; refusing to continue.", file=sys.stderr)
      for line_no, count, raw in bad_field_rows[:20]:
        print(f"  line {line_no}: fields={count} raw={raw}", file=sys.stderr)
      sys.exit(1)

    cleaned_map = {}
    bad_rows = []
    total = len(rows)
    total_chunks = (total + args.chunk_size - 1) // args.chunk_size

    for chunk_idx, start in enumerate(range(0, total, args.chunk_size), start=1):
      end = min(start + args.chunk_size, total)
      print(f"  chunk {chunk_idx}/{total_chunks}: lines {start + 1}-{end}", flush=True)
      process_range(filelist, rows, args.text_index, args.text_cleaners, start, end, cleaned_map, bad_rows)

    kept_rows = []
    for i, row in enumerate(rows):
      if i in cleaned_map:
        row[args.text_index] = cleaned_map[i]
        kept_rows.append(row)

    new_filelist = filelist + "." + args.out_extension
    with open(new_filelist, "w", encoding="utf-8") as f:
      f.writelines(["|".join(x) + "\n" for x in kept_rows])

    bad_path = filelist + "." + args.bad_extension
    with open(bad_path, "w", encoding="utf-8") as f:
      for item in bad_rows:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"  kept_rows: {len(kept_rows)} / {len(rows)}", flush=True)
    print(f"  bad_rows: {len(bad_rows)}", flush=True)
    print(f"  wrote: {new_filelist}", flush=True)
    print(f"  wrote: {bad_path}", flush=True)


if __name__ == "__main__":
  main()
