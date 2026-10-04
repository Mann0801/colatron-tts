"""Pretrained HiFi-GAN vocoder (SpeechBrain, trained on LJ Speech): mel -> waveform.

A GAN-trained convolutional network that upsamples each mel frame to 256 audio
samples in one parallel pass, so it is fast on CPU.
"""

from functools import lru_cache
from pathlib import Path

import torch
from speechbrain.inference.vocoders import HIFIGAN

import audio_config as cfg

SOURCE = "speechbrain/tts-hifigan-ljspeech"
SAVE_DIR = Path(__file__).resolve().parents[2] / "checkpoints" / "hifigan-ljspeech"


@lru_cache(maxsize=1)
def load_hifigan():
    return HIFIGAN.from_hparams(source=SOURCE, savedir=str(SAVE_DIR))


@torch.inference_mode()
def mel_to_wav(mel, mel_lengths=None):
    """mel: [1, 80, frames] log-magnitude. Returns (waveform as numpy, sample_rate)."""
    waveform = load_hifigan().decode_batch(mel)
    return waveform.squeeze().numpy(), cfg.SAMPLE_RATE
