"""Compare the mel spectrograms of the baseline and the main system on the same sentence.

Reads the committed sample audio, so no models or data are needed.

Usage:
    python src/plot_mel_comparison.py            # sentence 05
    python src/plot_mel_comparison.py --id 02
"""

import argparse
import csv
from pathlib import Path

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

import audio_config as cfg

SYSTEMS = [
    ("Baseline (ours, from scratch) + Griffin-Lim", "baseline"),
    ("Tacotron 2 + HiFi-GAN", "main_hifigan"),
]


def log_mel(path):
    wav, sr = sf.read(path)
    if sr != cfg.SAMPLE_RATE:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=cfg.SAMPLE_RATE)
    mel = librosa.feature.melspectrogram(
        y=wav, sr=cfg.SAMPLE_RATE, n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH,
        win_length=cfg.WIN_LENGTH, n_mels=cfg.N_MELS, fmin=cfg.F_MIN, fmax=cfg.F_MAX, power=1.0,
    )
    return np.log(np.maximum(mel, 1e-5))


def main():
    parser = argparse.ArgumentParser(description="Baseline vs main system mel spectrograms")
    parser.add_argument("--id", default="05")
    parser.add_argument("--out", default="samples/comparison/mel_baseline_vs_main.png")
    args = parser.parse_args()

    with open("samples/sentences.csv", newline="") as f:
        text = {row["id"]: row["text"] for row in csv.DictReader(f)}[args.id]

    mels = [log_mel(Path("samples") / folder / f"{args.id}.wav") for _, folder in SYSTEMS]
    vmin = min(m.min() for m in mels)
    vmax = max(m.max() for m in mels)
    seconds_per_frame = cfg.HOP_LENGTH / cfg.SAMPLE_RATE
    longest = max(m.shape[1] for m in mels) * seconds_per_frame

    fig, axes = plt.subplots(len(mels), 1, figsize=(10, 5.6), constrained_layout=True, sharex=True)
    for ax, mel, (label, _) in zip(axes, mels, SYSTEMS):
        ax.imshow(mel, origin="lower", aspect="auto", cmap="Purples", vmin=vmin, vmax=vmax,
                  extent=[0, mel.shape[1] * seconds_per_frame, 0, mel.shape[0]])
        ax.set_xlim(0, longest)
        ax.set_title(label, loc="left")
        ax.set_ylabel("Mel band")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f'"{text}"', fontsize=10)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
