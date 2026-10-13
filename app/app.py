"""Gradio demo: type text, pick a system, hear the speech, see the mel and attention.

The Colatron options need our trained weights at data/colatron/colatron.pt and the baseline
option needs data/baseline/baseline.pt (both gitignored; retrain with
python -m src.models.colatron train / python -m src.models.baseline train).

Usage:
    python app/app.py        then open http://127.0.0.1:7860
"""

import sys
import time
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from generate_samples import SENTENCES
from infer import VOCODERS, synthesize
from plot_alignment import alignment_stats, make_figure
from vocoder.hifigan import load_hifigan
from vocoder.griffinlim import load_griffinlim
from models.tacotron import load_tacotron, text_to_mel
import audio_config as cfg

VOCODER_LABELS = {
    "Tacotron 2 + HiFi-GAN (pretrained)": "hifigan",
    "Tacotron 2 + Griffin-Lim (no learning)": "griffinlim",
}
COLATRON_LABELS = {
    "Colatron (ours, trained from scratch) + HiFi-GAN": "hifigan",
    "Colatron (ours, trained from scratch) + Griffin-Lim": "griffinlim",
}
BASELINE_LABEL = "Baseline (ours, no attention) + Griffin-Lim"
MAX_CHARS = 300

_baseline = None
_colatron = None


def load_colatron():
    """Loads Colatron once. Raises a readable error if the weights are missing."""
    global _colatron
    if _colatron is None:
        from src.models.colatron import ColatronSynth
        try:
            _colatron = ColatronSynth()
        except FileNotFoundError:
            raise gr.Error("Colatron weights not found. Put colatron.pt in data/colatron/ "
                           "(or train it: python -m src.models.colatron train).")
    return _colatron


def colatron_tts(text, vocoder):
    import torch
    from src.models.colatron import alignment_stats as colatron_stats

    synth = load_colatron()
    start = time.time()
    mel, alignment = synth.mel(text)
    wav, sample_rate = VOCODERS[vocoder](torch.from_numpy(mel)[None])
    elapsed = time.time() - start

    plt.close("all")
    figure = make_figure(mel, alignment, text, model_name="Colatron (ours)")
    stats = colatron_stats(torch.from_numpy(alignment))
    duration = len(wav) / sample_rate
    info = (f"{duration:.2f}s of audio generated in {elapsed:.2f}s on CPU  |  "
            f"attention monotonic {stats['monotonic']:.0%}, coverage {stats['coverage']:.0%}")
    return (sample_rate, wav), figure, info


def load_baseline():
    """Loads the baseline once. Raises a readable error if the weights are missing."""
    global _baseline
    if _baseline is None:
        from src.models.baseline import BaselineSynth
        try:
            _baseline = BaselineSynth()
        except FileNotFoundError:
            raise gr.Error("Baseline weights not found. Put baseline.pt in data/baseline/ "
                           "(or train it: python -m src.models.baseline train).")
    return _baseline


def baseline_figure(baseline_mel, tacotron_mel, text):
    """Tacotron 2's mel on top, the baseline's below, on the same colour scale."""
    seconds_per_frame = cfg.HOP_LENGTH / cfg.SAMPLE_RATE
    mels = [("Tacotron 2 (pretrained, with attention)", tacotron_mel),
            ("Our baseline (no attention): timing is a fixed stretch, so the mel blurs", baseline_mel)]
    vmin = min(m.min() for _, m in mels)
    vmax = max(m.max() for _, m in mels)
    longest = max(m.shape[1] for _, m in mels) * seconds_per_frame

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True, sharex=True)
    for ax, (title, mel) in zip(axes, mels):
        ax.imshow(mel, origin="lower", aspect="auto", cmap="Purples", vmin=vmin, vmax=vmax,
                  extent=[0, mel.shape[1] * seconds_per_frame, 0, mel.shape[0]])
        ax.set_xlim(0, longest)
        ax.set_title(title, loc="left")
        ax.set_ylabel("Mel band")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f'"{text}"', fontsize=10, wrap=True)
    return fig


def baseline_tts(text):
    from src.preprocess.common import mel_to_audio

    synth = load_baseline()
    start = time.time()
    mel = synth.mel(text)
    wav, sample_rate = mel_to_audio(mel)
    elapsed = time.time() - start

    tacotron_mel, _, _ = text_to_mel(text)
    plt.close("all")
    figure = baseline_figure(mel, tacotron_mel[0].numpy(), text)
    duration = len(wav) / sample_rate
    info = (f"{duration:.2f}s of audio generated in {elapsed:.2f}s on CPU  |  "
            f"no attention, so no alignment to show")
    return (sample_rate, wav), figure, info


def tts(text, vocoder_label):
    text = text.strip()
    if not text:
        raise gr.Error("Please type a sentence.")
    if len(text) > MAX_CHARS:
        raise gr.Error(f"Please keep it under {MAX_CHARS} characters.")
    if vocoder_label == BASELINE_LABEL:
        return baseline_tts(text)
    if vocoder_label in COLATRON_LABELS:
        return colatron_tts(text, COLATRON_LABELS[vocoder_label])

    start = time.time()
    wav, sample_rate, mel, alignment = synthesize(text, VOCODER_LABELS[vocoder_label])
    elapsed = time.time() - start

    plt.close("all")
    figure = make_figure(mel[0].numpy(), alignment[0].numpy(), text)
    stats = alignment_stats(alignment[0])
    duration = len(wav) / sample_rate
    info = (f"{duration:.2f}s of audio generated in {elapsed:.2f}s on CPU  |  "
            f"attention monotonic {stats['monotonic']:.0%}, coverage {stats['coverage']:.0%}")
    return (sample_rate, wav), figure, info


def build_app():
    with gr.Blocks(title="Colatron TTS") as demo:
        gr.Markdown(
            "# Colatron: end-to-end text to speech\n"
            "Text → acoustic model → mel spectrogram → **vocoder** → waveform.  \n"
            "**Colatron** is our seq2seq model with attention, trained from scratch on LJ Speech. "
            "**Tacotron 2** (pretrained) is the reference. The **baseline** has no attention: "
            "listen to it to hear why attention matters."
        )
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label="Text", lines=3, value=SENTENCES[3],
                                  max_length=MAX_CHARS)
                vocoder = gr.Radio(list(COLATRON_LABELS) + list(VOCODER_LABELS) + [BASELINE_LABEL],
                                   value=next(iter(COLATRON_LABELS)), label="System")
                button = gr.Button("Speak", variant="primary")
                gr.Examples([[s] for s in SENTENCES[:6]], inputs=[text])
            with gr.Column():
                audio = gr.Audio(label="Generated speech", type="numpy", autoplay=True)
                info = gr.Markdown()
        plot = gr.Plot(label="Mel spectrogram and attention alignment (baseline: mel vs Tacotron 2)")

        button.click(tts, inputs=[text, vocoder], outputs=[audio, plot, info])
        text.submit(tts, inputs=[text, vocoder], outputs=[audio, plot, info])
    return demo


if __name__ == "__main__":
    print("Loading models...")
    load_tacotron(), load_griffinlim(), load_hifigan()
    assert set(VOCODER_LABELS.values()) == set(VOCODERS)
    build_app().launch()
