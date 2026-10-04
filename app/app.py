"""Gradio demo: type text, pick a vocoder, hear the speech, see the mel and attention.

Usage:
    python app/app.py        then open http://127.0.0.1:7860
"""

import sys
import time
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from generate_samples import SENTENCES
from infer import VOCODERS, synthesize
from plot_alignment import alignment_stats, make_figure
from vocoder.hifigan import load_hifigan
from vocoder.griffinlim import load_griffinlim
from models.tacotron import load_tacotron

VOCODER_LABELS = {
    "HiFi-GAN (neural, pretrained)": "hifigan",
    "Griffin-Lim (signal processing, no learning)": "griffinlim",
}
MAX_CHARS = 300


def tts(text, vocoder_label):
    text = text.strip()
    if not text:
        raise gr.Error("Please type a sentence.")
    if len(text) > MAX_CHARS:
        raise gr.Error(f"Please keep it under {MAX_CHARS} characters.")

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
            "Text → **Tacotron 2** (pretrained on LJ Speech) → mel spectrogram → "
            "**vocoder** → waveform. Pick a vocoder to compare them on the same pipeline."
        )
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label="Text", lines=3, value=SENTENCES[3],
                                  max_length=MAX_CHARS)
                vocoder = gr.Radio(list(VOCODER_LABELS), value=next(iter(VOCODER_LABELS)),
                                   label="Vocoder")
                button = gr.Button("Speak", variant="primary")
                gr.Examples([[s] for s in SENTENCES[:6]], inputs=[text])
            with gr.Column():
                audio = gr.Audio(label="Generated speech", type="numpy", autoplay=True)
                info = gr.Markdown()
        plot = gr.Plot(label="Mel spectrogram and attention alignment")

        button.click(tts, inputs=[text, vocoder], outputs=[audio, plot, info])
        text.submit(tts, inputs=[text, vocoder], outputs=[audio, plot, info])
    return demo


if __name__ == "__main__":
    print("Loading models...")
    load_tacotron(), load_griffinlim(), load_hifigan()
    assert set(VOCODER_LABELS.values()) == set(VOCODERS)
    build_app().launch()
