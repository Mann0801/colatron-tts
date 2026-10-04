"""Griffin-Lim vocoder: mel spectrogram -> waveform with no neural network.

Step 1: undo the mel filterbank to get an approximate linear spectrogram.
Step 2: Griffin-Lim iteratively estimates the missing phase so that the
        STFT of the waveform matches that spectrogram.
"""

from functools import lru_cache

import torch
from torchaudio.transforms import GriffinLim, InverseMelScale

import audio_config as cfg


@lru_cache(maxsize=1)
def load_griffinlim(n_iter=60):
    inverse_mel = InverseMelScale(
        n_stft=cfg.N_FFT // 2 + 1,
        n_mels=cfg.N_MELS,
        sample_rate=cfg.SAMPLE_RATE,
        f_min=cfg.F_MIN,
        f_max=cfg.F_MAX,
        mel_scale="slaney",
        norm="slaney",
    )
    griffin_lim = GriffinLim(
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
        win_length=cfg.WIN_LENGTH,
        power=1.0,
        n_iter=n_iter,
    )
    return inverse_mel, griffin_lim


@torch.inference_mode()
def mel_to_wav(mel, mel_lengths=None):
    """mel: [1, 80, frames] log-magnitude. Returns (waveform as numpy, sample_rate)."""
    inverse_mel, griffin_lim = load_griffinlim()
    magnitude_mel = torch.exp(mel)
    linear_spec = inverse_mel(magnitude_mel)
    waveform = griffin_lim(linear_spec)
    return waveform[0].numpy(), cfg.SAMPLE_RATE
