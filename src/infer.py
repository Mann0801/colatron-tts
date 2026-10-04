"""Text-to-speech from the command line.

Usage:
    python src/infer.py --text "Hello world" --out outputs/hello.wav
"""

import argparse
import time
from pathlib import Path

import soundfile as sf

from models.tacotron import text_to_mel
from vocoder import griffinlim, hifigan

VOCODERS = {
    "griffinlim": griffinlim.mel_to_wav,
    "hifigan": hifigan.mel_to_wav,
}


def synthesize(text, vocoder="griffinlim"):
    mel, mel_lengths, alignment = text_to_mel(text)
    wav, sample_rate = VOCODERS[vocoder](mel, mel_lengths)
    return wav, sample_rate, mel, alignment


def main():
    parser = argparse.ArgumentParser(description="Text -> Tacotron 2 -> vocoder -> .wav")
    parser.add_argument("--text", required=True)
    parser.add_argument("--vocoder", choices=VOCODERS, default="griffinlim")
    parser.add_argument("--out", default="outputs/out.wav")
    args = parser.parse_args()

    start = time.time()
    wav, sample_rate, mel, alignment = synthesize(args.text, args.vocoder)
    elapsed = time.time() - start

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, wav, sample_rate)

    duration = len(wav) / sample_rate
    print(f"text      : {args.text}")
    print(f"mel       : {tuple(mel.shape)}  alignment: {tuple(alignment.shape)}")
    print(f"audio     : {duration:.2f}s at {sample_rate} Hz")
    print(f"time      : {elapsed:.1f}s ({args.vocoder})")
    print(f"saved to  : {out}")


if __name__ == "__main__":
    main()
