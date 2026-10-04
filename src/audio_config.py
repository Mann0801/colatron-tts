"""Mel spectrogram settings shared by every part of the project.

These match the pretrained Tacotron 2 and HiFi-GAN (both trained on LJ Speech),
so preprocessing, the baseline, and all vocoders read and write the same format.
Mels are stored as natural log of magnitude: log(clamp(mel, min=1e-5)).
"""

SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_MELS = 80
F_MIN = 0.0
F_MAX = 8000.0
