"""Clean the text, compute mels and make the train/val/test split.

    python -m src.preprocess.prepare

Reads  data/raw/metadata.csv + data/raw/wavs/*.wav        (made by download.py)
Writes data/processed/mels/<id>.npy      (n_mels, frames) natural-log mel
       data/processed/metadata.csv       id|raw_text|clean_text|n_frames
       data/processed/vocab.json         the fixed character -> id table
       data/processed/splits.json        train / val / test ids
       data/processed/mel_config.json    mel settings that were used
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.preprocess.audio_features import load_wav, wav_to_mel  # noqa: E402
from src.preprocess.common import MEL_DIR, PROC_DIR, RAW_DIR, RAW_WAVS, audio_config  # noqa: E402
from src.preprocess.text_cleaning import SYMBOLS, clean_text  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--min-seconds", type=float, default=0.5)
    args = ap.parse_args()

    cfg = audio_config()
    print("Mel settings:", cfg)

    raw_meta = RAW_DIR / "metadata.csv"
    if not raw_meta.exists():
        sys.exit(f"{raw_meta} not found. Run first: python -m src.preprocess.download")

    rows = []
    with open(raw_meta, encoding="utf-8") as f:
        next(f, None)
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) >= 2:
                rows.append((parts[0], parts[1]))

    MEL_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for i, (clip_id, raw) in enumerate(rows, start=1):
        wav_path = RAW_WAVS / f"{clip_id}.wav"
        clean = clean_text(raw)
        if not wav_path.exists() or not clean:
            print(f"  skipping {clip_id} (missing wav or empty text)")
            continue
        y = load_wav(wav_path, cfg["sr"])
        if len(y) / cfg["sr"] < args.min_seconds:
            print(f"  skipping {clip_id} (too short)")
            continue
        mel = wav_to_mel(y, cfg)
        np.save(MEL_DIR / f"{clip_id}.npy", mel)
        records.append((clip_id, raw, clean, mel.shape[1]))
        if i % 50 == 0:
            print(f"  processed {i}/{len(rows)}", flush=True)

    if len(records) < 3:
        sys.exit("Fewer than 3 usable clips; cannot make a train/val/test split.")

    with open(PROC_DIR / "metadata.csv", "w", encoding="utf-8") as f:
        f.write("id|raw_text|clean_text|n_frames\n")
        for clip_id, raw, clean, n_frames in records:
            f.write(f"{clip_id}|{raw}|{clean}|{n_frames}\n")

    with open(PROC_DIR / "vocab.json", "w", encoding="utf-8") as f:
        json.dump({"symbols": SYMBOLS, "note": "id = position in this list"}, f, indent=2)
    with open(PROC_DIR / "mel_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    ids = sorted(r[0] for r in records)
    random.Random(args.seed).shuffle(ids)
    n_test = max(1, round(len(ids) * args.test_frac))
    n_val = max(1, round(len(ids) * args.val_frac))
    splits = {
        "test": sorted(ids[:n_test]),
        "val": sorted(ids[n_test:n_test + n_val]),
        "train": sorted(ids[n_test + n_val:]),
    }
    with open(PROC_DIR / "splits.json", "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    total_h = sum(r[3] for r in records) * cfg["hop"] / cfg["sr"] / 3600
    print(f"Done: {len(records)} clips ({total_h:.2f} h) -> "
          f"train {len(splits['train'])}, val {len(splits['val'])}, test {len(splits['test'])}")
    print(f"Saved to {PROC_DIR}")


if __name__ == "__main__":
    main()
