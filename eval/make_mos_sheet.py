"""Build the listening-test material for the Google Form.

    python eval/make_mos_sheet.py            # 6 sentences x 3 systems = 18 clips

Systems: our baseline, our HiFi-GAN system, and the real recording.
Writes to eval/mos/:
    audio/clip_01.wav ...   anonymous, shuffled, loudness-matched clips (upload these to the form)
    mos_sheet.csv           order, file, sentence   (share this one)
    mos_key.csv             file -> system          (KEEP PRIVATE; needed to score the ratings)
"""
import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocess.audio_features import load_wav  # noqa: E402
from src.preprocess.common import (  # noqa: E402
    RAW_WAVS, ROOT, audio_config, import_first, read_metadata, read_splits, resample_to, save_wav, to_numpy,
)

SYSTEMS = {"baseline": "Baseline (ours)", "hifigan": "HiFi-GAN system (ours)", "real": "Real recording"}


def match_loudness(wav, target_rms=0.06):
    wav = np.asarray(wav, dtype=np.float32)
    rms = float(np.sqrt(np.mean(wav ** 2))) + 1e-8
    wav = wav * (target_rms / rms)
    peak = float(np.abs(wav).max())
    return wav * (0.95 / peak) if peak > 0.95 else wav


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=6, help="number of sentences (5-8)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "eval" / "mos"))
    args = ap.parse_args()

    from src.models.baseline import BaselineSynth  # imports torch, so do it here

    cfg = audio_config()
    infer = import_first("infer", "src.infer")
    baseline = BaselineSynth()
    meta = read_metadata()
    test_ids = read_splits()["test"]

    rng = random.Random(args.seed)
    medium = [i for i in test_ids if 40 <= len(meta[i]["raw"]) <= 160]
    pool = medium if len(medium) >= args.n else test_ids
    chosen = rng.sample(pool, min(args.n, len(pool)))

    items = []  # (system, text, wav)
    for clip_id in chosen:
        text = meta[clip_id]["raw"]
        print(f"{clip_id}: {text[:70]}")
        items.append(("real", text, load_wav(RAW_WAVS / f"{clip_id}.wav", cfg["sr"])))

        wav, sr = baseline.wav(text)
        items.append(("baseline", text, resample_to(wav, sr, cfg["sr"])))

        out = infer.synthesize(text, "hifigan")
        wav, sr = to_numpy(out[0]).astype(np.float32).squeeze(), int(out[1])
        items.append(("hifigan", text, resample_to(wav, sr, cfg["sr"])))

    rng.shuffle(items)
    out_dir = Path(args.out)
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "mos_sheet.csv", "w", newline="", encoding="utf-8") as fs, \
            open(out_dir / "mos_key.csv", "w", newline="", encoding="utf-8") as fk:
        sheet, key = csv.writer(fs), csv.writer(fk)
        sheet.writerow(["order", "file", "sentence"])
        key.writerow(["file", "system"])
        for n, (system, text, wav) in enumerate(items, start=1):
            name = f"clip_{n:02d}.wav"
            save_wav(out_dir / "audio" / name, match_loudness(wav), cfg["sr"])
            sheet.writerow([n, name, text])
            key.writerow([name, system])

    print(f"\nWrote {len(items)} clips to {out_dir / 'audio'}")
    print(f"Share:   {out_dir / 'mos_sheet.csv'}")
    print(f"Private: {out_dir / 'mos_key.csv'}  (maps clips to systems)")


if __name__ == "__main__":
    main()
