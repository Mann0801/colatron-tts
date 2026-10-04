"""Pretrained WaveRNN vocoder (trained on LJ Speech): mel spectrogram -> waveform."""

from functools import lru_cache

import torch

from models.tacotron import BUNDLE


@lru_cache(maxsize=1)
def load_wavernn():
    return BUNDLE.get_vocoder().eval()


@torch.inference_mode()
def mel_to_wav(mel, mel_lengths):
    """Returns (waveform as a 1-D numpy array, sample_rate)."""
    vocoder = load_wavernn()
    waveform, _ = vocoder(mel, mel_lengths)
    return waveform[0].numpy(), vocoder.sample_rate
