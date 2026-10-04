"""Objective score: mel distance between generated and real audio on the test clips.

    python eval/objective.py              # first 20 test clips
    python eval/objective.py --n 50

For each test sentence the audio of each system is turned into a mel with the SAME
settings as the real clip, the two mels are aligned with DTW (the generated clip may
have a different length), and we average the per-frame distance along the path:
    Mel L1   = mean absolute difference of the log-mel values
    Mel RMSE = root mean squared difference of the log-mel values
Lower is better. Writes eval/results/objective.csv and objective_per_clip.csv.
"""
import argparse
import csv
import sys
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocess.audio_features import wav_to_mel  # noqa: E402
from src.preprocess.common import (  # noqa: E402
    MEL_DIR, ROOT, audio_config, import_first, read_metadata, read_splits, resample_to, to_numpy,
)


def dtw_mel_distance(real, gen):
    """Both (n_mels, frames). Returns (L1, RMSE) averaged along the DTW alignment path."""
    _, path = librosa.sequence.dtw(X=real, Y=gen, metric="euclidean")
    i, j = path[:, 0], path[:, 1]
    diff = real[:, i] - gen[:, j]
    return float(np.abs(diff).mean()), float(np.sqrt((diff ** 2).mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20, help="number of test clips to score")
    ap.add_argument("--out", default=str(ROOT / "eval" / "results"))
    args = ap.parse_args()

    from src.models.baseline import BaselineSynth  # imports torch, so do it here

    cfg = audio_config()
    infer = import_first("infer", "src.infer")
    baseline = BaselineSynth()
    meta = read_metadata()
    test_ids = read_splits()["test"][:args.n]

    per_clip = []  # (clip_id, system, l1, rmse)
    for k, clip_id in enumerate(test_ids, start=1):
        text = meta[clip_id]["raw"]
        real = np.load(MEL_DIR / f"{clip_id}.npy")

        wav, sr = baseline.wav(text)
        gen = wav_to_mel(resample_to(wav, sr, cfg["sr"]), cfg)
        per_clip.append((clip_id, "baseline") + dtw_mel_distance(real, gen))

        out = infer.synthesize(text, "hifigan")
        wav, sr = to_numpy(out[0]).astype(np.float32).squeeze(), int(out[1])
        gen = wav_to_mel(resample_to(wav, sr, cfg["sr"]), cfg)
        per_clip.append((clip_id, "hifigan") + dtw_mel_distance(real, gen))
        print(f"  scored {k}/{len(test_ids)}", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "objective_per_clip.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clip", "system", "mel_l1", "mel_rmse"])
        w.writerows(per_clip)

    summary = []
    for system in ("baseline", "hifigan"):
        rows = [r for r in per_clip if r[1] == system]
        summary.append((system, float(np.mean([r[2] for r in rows])), float(np.mean([r[3] for r in rows])), len(rows)))
    with open(out_dir / "objective.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "mel_l1", "mel_rmse", "n_clips"])
        for system, l1, rmse, n in summary:
            w.writerow([system, f"{l1:.4f}", f"{rmse:.4f}", n])

    print("\nObjective mel distance (DTW-aligned, lower is better)")
    print(f"{'System':<26}{'Mel L1':>10}{'Mel RMSE':>12}{'Clips':>8}")
    names = {"baseline": "Baseline (ours)", "hifigan": "HiFi-GAN system (ours)"}
    for system, l1, rmse, n in summary:
        print(f"{names[system]:<26}{l1:>10.3f}{rmse:>12.3f}{n:>8}")
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
