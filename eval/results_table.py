"""Final results table: our systems next to the paper's MOS numbers.

    python eval/results_table.py --ratings path/to/form_responses.csv

Run eval/make_mos_sheet.py and eval/objective.py first. The ratings file can be either
  * the Google Form export, where each rating question is titled with its clip name
    (e.g. "clip_07"), one row per respondent, answers 1-5; or
  * a long CSV with the columns: file,rating   (e.g. clip_07.wav,4)
Without --ratings the MOS cells show "pending".
Writes eval/results/results_table.md.
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

PAPER_MOS = [("SVR (paper)", 1.0), ("NN (paper)", 1.7), ("Seq2seq (paper)", 2.5), ("Tacotron (paper)", 3.82)]
OURS = [("Baseline (ours)", "baseline"), ("HiFi-GAN system (ours)", "hifigan"), ("Real recording", "real")]


def load_ratings(path):
    """Returns list of (clip_file, rating)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if not rows:
        return []
    header = [h.strip().lower() for h in rows[0]]
    if "file" in header and "rating" in header:
        fi, ri = header.index("file"), header.index("rating")
        out = []
        for r in rows[1:]:
            if len(r) > max(fi, ri) and re.match(r"\s*[1-5]", r[ri]):
                name = r[fi].strip()
                name = name if name.endswith(".wav") else name + ".wav"
                out.append((name, int(re.match(r"\s*([1-5])", r[ri]).group(1))))
        return out

    columns = {}
    for idx, h in enumerate(rows[0]):
        m = re.search(r"clip_\d+", h, flags=re.IGNORECASE)
        if m:
            columns[idx] = m.group(0).lower() + ".wav"
    out = []
    for r in rows[1:]:
        for idx, name in columns.items():
            if idx < len(r):
                m = re.match(r"\s*([1-5])", r[idx])
                if m:
                    out.append((name, int(m.group(1))))
    return out


def mos_by_system(ratings_path, key_path):
    with open(key_path, newline="", encoding="utf-8") as f:
        key = {r["file"]: r["system"] for r in csv.DictReader(f)}
    scores = {}
    for name, rating in load_ratings(ratings_path):
        if name in key:
            scores.setdefault(key[name], []).append(rating)
    result = {}
    for system, vals in scores.items():
        vals = np.array(vals, dtype=float)
        ci = 1.96 * vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else float("nan")
        result[system] = (vals.mean(), ci, len(vals))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ratings", default=None)
    ap.add_argument("--key", default=str(ROOT / "eval" / "mos" / "mos_key.csv"))
    ap.add_argument("--objective", default=str(ROOT / "eval" / "results" / "objective.csv"))
    ap.add_argument("--out", default=str(ROOT / "eval" / "results" / "results_table.md"))
    args = ap.parse_args()

    mos = {}
    if args.ratings:
        if not Path(args.key).exists():
            sys.exit(f"{args.key} not found. Run eval/make_mos_sheet.py first.")
        mos = mos_by_system(args.ratings, args.key)

    objective = {}
    if Path(args.objective).exists():
        with open(args.objective, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                objective[r["system"]] = (float(r["mel_l1"]), float(r["mel_rmse"]))
    else:
        print(f"[note] {args.objective} not found; run eval/objective.py for the mel distance columns")

    lines = ["| System | MOS (1-5) | Mel L1 | Mel RMSE |", "|---|---|---|---|"]
    for name, value in PAPER_MOS:
        lines.append(f"| {name} | {value} | - | - |")
    for name, system in OURS:
        if system in mos:
            mean, ci, n = mos[system]
            cell = f"{mean:.2f} +/- {ci:.2f} (n={n})" if n > 1 else f"{mean:.2f} (n={n})"
        else:
            cell = "pending"
        l1, rmse = objective.get(system, (None, None))
        l1_cell = f"{l1:.3f}" if l1 is not None else "-"
        rmse_cell = f"{rmse:.3f}" if rmse is not None else "-"
        lines.append(f"| {name} | {cell} | {l1_cell} | {rmse_cell} |")

    table = "\n".join(lines)
    print(table)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(table + "\n", encoding="utf-8")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
