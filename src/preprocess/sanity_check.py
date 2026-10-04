"""Sanity check: plot one real mel and turn it back into audio with Griffin-Lim.

    python -m src.preprocess.sanity_check            # uses the first test clip
    python -m src.preprocess.sanity_check --id LJ001-0010

Writes to eval/outputs/sanity/: <id>_mel.png, <id>_original.wav, <id>_griffinlim.wav
Listen to the two wavs: the Griffin-Lim one will sound a bit robotic, but the words must
be clear. If it is noise, the mel settings do not match the vocoder.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.preprocess.audio_features import load_wav, wav_to_mel  # noqa: E402
from src.preprocess.common import (  # noqa: E402
    MEL_DIR, RAW_WAVS, ROOT, audio_config, mel_to_audio, read_metadata, read_splits, save_wav,
)

OUT_DIR = ROOT / "eval" / "outputs" / "sanity"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", default=None, help="clip id (default: first test clip)")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = audio_config()
    meta = read_metadata()
    clip_id = args.id or read_splits()["test"][0]
    if clip_id not in meta:
        sys.exit(f"{clip_id} is not in data/processed/metadata.csv")
    mel = np.load(MEL_DIR / f"{clip_id}.npy")
    print(f"Clip {clip_id}: {meta[clip_id]['raw']}")
    print(f"Mel shape (bands, frames): {mel.shape}, range {mel.min():.2f} to {mel.max():.2f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) plot
    fig, ax = plt.subplots(figsize=(10, 3.5))
    seconds = mel.shape[1] * cfg["hop"] / cfg["sr"]
    im = ax.imshow(mel, origin="lower", aspect="auto", interpolation="none",
                   extent=[0, seconds, 0, cfg["fmax"]])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    ax.set_title(f"{clip_id}: log-mel spectrogram ({mel.shape[0]} bands)")
    fig.colorbar(im, ax=ax, label="natural log magnitude")
    fig.tight_layout()
    png = OUT_DIR / f"{clip_id}_mel.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)

    # 2) mel -> audio with Griffin-Lim, plus the original for comparison
    original = load_wav(RAW_WAVS / f"{clip_id}.wav", cfg["sr"])
    save_wav(OUT_DIR / f"{clip_id}_original.wav", original, cfg["sr"])
    wav, sr = mel_to_audio(mel)
    save_wav(OUT_DIR / f"{clip_id}_griffinlim.wav", wav, sr)

    # 3) numeric round-trip check: mel(resynthesised audio) vs the original mel
    if sr != cfg["sr"]:
        print(f"[note] vocoder returned sr={sr}, config sr={cfg['sr']}")
    again = wav_to_mel(wav, cfg)
    frames = min(again.shape[1], mel.shape[1])
    l1 = float(np.abs(again[:, :frames] - mel[:, :frames]).mean())
    print(f"Round-trip mel L1 distance: {l1:.3f}  (lower is better; judge by listening too)")
    print(f"Saved plot and audio to {OUT_DIR}")


if __name__ == "__main__":
    main()
