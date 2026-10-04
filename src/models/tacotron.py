"""Pretrained Tacotron 2 (trained on LJ Speech): text -> mel spectrogram."""

from functools import lru_cache

import torch
import torchaudio

BUNDLE = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH


@lru_cache(maxsize=1)
def load_tacotron():
    processor = BUNDLE.get_text_processor()
    model = BUNDLE.get_tacotron2().eval()
    return processor, model


@torch.inference_mode()
def text_to_mel(text):
    """Returns (mel [1, 80, frames], mel_lengths [1], alignment [1, frames, chars])."""
    processor, model = load_tacotron()
    tokens, lengths = processor(text)
    mel, mel_lengths, alignment = model.infer(tokens, lengths)
    return mel, mel_lengths, alignment
