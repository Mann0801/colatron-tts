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
from src.preprocess.audio_features import load_wav, wav_to_mel  # noqa: E402
from src.preprocess.common import (  # noqa: E402
    MEL_DIR, ROOT, audio_config, import_first, read_metadata, read_splits, resample_to, to_numpy,
)


def dtw_mel_distance(real, gen):
    """Both (n_mels, frames). Returns (L1, RMSE) averaged along the DTW alignment path."""
    _, path = librosa.sequence.dtw(X=real, Y=gen, metric="euclidean")
    i, j = path[:, 0], path[:, 1]
    diff = real[:, i] - gen[:, j]
    return float(np.abs(diff).mean()), float(np.sqrt((diff ** 2).mean()))


SYSTEMS = [  # (key, display name)
    ("baseline", "Baseline (ours) + Griffin-Lim"),
    ("colatron_griffinlim", "Colatron (ours) + Griffin-Lim"),
    ("colatron_hifigan", "Colatron (ours) + HiFi-GAN"),
    ("tacotron_griffinlim", "Tacotron 2 + Griffin-Lim"),
    ("hifigan", "Tacotron 2 + HiFi-GAN"),
]
FULL_DIR = ROOT / "data" / "full"


def load_test_set(n):
    """Test ids, raw transcripts and a function returning the real mel of a clip.

    Prefers data/full/ (prepare_full.py: same 40 test clips, mel recomputed from the wav),
    and falls back to the 404-clip subset in data/processed/ (prepare.py)."""
    if (FULL_DIR / "splits.json").exists():
        import json

        ids = json.loads((FULL_DIR / "splits.json").read_text())["test"][:n]
        with open(FULL_DIR / "metadata.csv", encoding="utf-8") as f:
            next(f)
            texts = dict(line.rstrip("\n").split("|", 1) for line in f)
        cfg = audio_config()
        return ids, texts, lambda i: wav_to_mel(load_wav(FULL_DIR / "wavs" / f"{i}.wav", cfg["sr"]), cfg)
    meta = read_metadata()
    ids = read_splits()["test"][:n]
    return ids, {i: meta[i]["raw"] for i in ids}, lambda i: np.load(MEL_DIR / f"{i}.npy")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20, help="number of test clips to score")
    ap.add_argument("--out", default=str(ROOT / "eval" / "results"))
    args = ap.parse_args()

    import torch  # imports torch, so do it here
    from src.models.baseline import BaselineSynth
    from src.models.colatron import ColatronSynth

    cfg = audio_config()
    infer = import_first("infer", "src.infer")
    gl = import_first("vocoder.griffinlim", "src.vocoder.griffinlim")
    hifi = import_first("vocoder.hifigan", "src.vocoder.hifigan")
    baseline = BaselineSynth()
    try:
        colatron = ColatronSynth()
    except FileNotFoundError as e:
        colatron = None
        print(f"[note] skipping Colatron: {e}")
    test_ids, texts, real_mel = load_test_set(args.n)

    def score(clip_id, system, wav, sr, real):
        wav = to_numpy(wav).astype(np.float32).squeeze()
        gen = wav_to_mel(resample_to(wav, int(sr), cfg["sr"]), cfg)
        per_clip.append((clip_id, system) + dtw_mel_distance(real, gen))

    per_clip = []  # (clip_id, system, l1, rmse)
    for k, clip_id in enumerate(test_ids, start=1):
        text = texts[clip_id]
        real = real_mel(clip_id)

        score(clip_id, "baseline", *baseline.wav(text), real)

        torch.manual_seed(0)  # Tacotron 2's prenet dropout is random: same mel for both vocoders
        mel, mel_lengths, _ = infer.text_to_mel(text)
        score(clip_id, "tacotron_griffinlim", *gl.mel_to_wav(mel, mel_lengths), real)
        score(clip_id, "hifigan", *hifi.mel_to_wav(mel, mel_lengths), real)

        if colatron is not None:
            mel, _ = colatron.mel(text)
            mel = torch.from_numpy(mel)[None]
            score(clip_id, "colatron_griffinlim", *gl.mel_to_wav(mel), real)
            score(clip_id, "colatron_hifigan", *hifi.mel_to_wav(mel), real)
        print(f"  scored {k}/{len(test_ids)}", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "objective_per_clip.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clip", "system", "mel_l1", "mel_rmse"])
        w.writerows(per_clip)

    summary = []
    for system, _ in SYSTEMS:
        rows = [r for r in per_clip if r[1] == system]
        if rows:
            summary.append((system, float(np.mean([r[2] for r in rows])),
                            float(np.mean([r[3] for r in rows])), len(rows)))
    with open(out_dir / "objective.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "mel_l1", "mel_rmse", "n_clips"])
        for system, l1, rmse, n in summary:
            w.writerow([system, f"{l1:.4f}", f"{rmse:.4f}", n])

    print("\nObjective mel distance (DTW-aligned, lower is better)")
    print(f"{'System':<32}{'Mel L1':>10}{'Mel RMSE':>12}{'Clips':>8}")
    names = dict(SYSTEMS)
    for system, l1, rmse, n in summary:
        print(f"{names[system]:<32}{l1:>10.3f}{rmse:>12.3f}{n:>8}")
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
