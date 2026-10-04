"""wav -> 80-band log-mel spectrogram, using the team's settings from src/audio_config.py.

Output shape is (n_mels, frames), natural log, float32 (same layout the vocoders expect).
"""
import sys
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.preprocess.common import audio_config  # noqa: E402

LOG_FLOOR = 1e-5  # avoids log(0)


def load_wav(path, sr=None):
    sr = sr or audio_config()["sr"]
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32)


def wav_to_mel(y, cfg=None):
    cfg = cfg or audio_config()
    mel = librosa.feature.melspectrogram(
        y=np.asarray(y, dtype=np.float32),
        sr=cfg["sr"],
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop"],
        win_length=cfg["win"],
        window="hann",
        center=True,
        pad_mode="reflect",
        power=cfg["power"],
        n_mels=cfg["n_mels"],
        fmin=cfg["fmin"],
        fmax=cfg["fmax"],
        htk=False,
        norm="slaney",
    )
    return np.log(np.maximum(mel, LOG_FLOOR)).astype(np.float32)
