"""Generate the fixed test sentences through both vocoders into samples/.

Each sentence is turned into a mel spectrogram ONCE, and that same mel is sent
to Griffin-Lim and to HiFi-GAN, so any difference you hear comes from the vocoder.
A fixed seed per sentence makes the output reproducible (Tacotron 2 keeps
dropout on in its prenet at inference, so it is slightly random otherwise).

Usage:
    python src/generate_samples.py
"""

import csv
import time
from pathlib import Path

import soundfile as sf
import torch

from models.tacotron import text_to_mel
from vocoder import griffinlim, hifigan

SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts.",
    "Machine learning lets a computer learn patterns from examples instead of following fixed rules.",
    "Welcome to our mini project on end to end speech synthesis.",
    "She sells sea shells by the sea shore.",
    "The weather today is sunny, with a gentle breeze in the afternoon.",
    "Please remember to submit your assignment before the deadline on Saturday.",
    "A neural vocoder turns a mel spectrogram into a natural sounding waveform.",
    "Can you believe how quickly the semester has gone by?",
    "The commission concluded that the evidence was not sufficient to support the claim.",
]

VOCODERS = {"griffinlim": griffinlim.mel_to_wav, "hifigan": hifigan.mel_to_wav}
OUT_DIR = Path("samples")
SEED = 1234


def main():
    rows = []
    for i, text in enumerate(SENTENCES, start=1):
        torch.manual_seed(SEED + i)
        mel, mel_lengths, _ = text_to_mel(text)

        row = {"id": f"{i:02d}", "text": text}
        for name, mel_to_wav in VOCODERS.items():
            start = time.time()
            wav, sample_rate = mel_to_wav(mel, mel_lengths)
            elapsed = time.time() - start

            out_path = OUT_DIR / f"main_{name}" / f"{i:02d}.wav"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out_path, wav, sample_rate)
            row[f"{name}_seconds"] = f"{elapsed:.2f}"

        row["audio_seconds"] = f"{len(wav) / sample_rate:.2f}"
        rows.append(row)
        print(f"[{row['id']}] {row['audio_seconds']}s audio | "
              f"griffinlim {row['griffinlim_seconds']}s | hifigan {row['hifigan_seconds']}s | {text}")

    manifest = OUT_DIR / "sentences.csv"
    with open(manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} sentences x {len(VOCODERS)} vocoders, manifest: {manifest}")


if __name__ == "__main__":
    main()
