"""Plot Tacotron 2's mel spectrogram and attention alignment for a few sentences.

A clean diagonal in the alignment means the decoder reads the text in order,
one character after another, without skipping or repeating words.

Usage:
    python src/plot_alignment.py                      # default sentences
    python src/plot_alignment.py --text "Hello world"
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import audio_config as cfg
from models.tacotron import text_to_mel

DEFAULT_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts.",
    "Machine learning lets a computer learn patterns from examples instead of following fixed rules.",
]


def alignment_stats(alignment):
    """How well the attention follows the text.

    monotonic: share of decoder steps where the focused character does not move backwards.
    coverage:  share of input characters that were ever the main focus.
    focus:     average attention weight on the main character (1.0 = perfectly sharp).
    """
    weights, focused = alignment.max(dim=1)
    steps_forward = (focused[1:] >= focused[:-1]).float().mean().item()
    coverage = focused.unique().numel() / alignment.shape[1]
    return {"monotonic": steps_forward, "coverage": coverage, "focus": weights.mean().item()}


def make_figure(mel, alignment, text):
    """mel: [80, frames], alignment: [frames, chars]. Returns a matplotlib Figure."""
    seconds_per_frame = cfg.HOP_LENGTH / cfg.SAMPLE_RATE
    duration = mel.shape[1] * seconds_per_frame

    fig, (ax_mel, ax_att) = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)

    ax_mel.imshow(mel, origin="lower", aspect="auto", cmap="Purples",
                  extent=[0, duration, 0, mel.shape[0]])
    ax_mel.set_title("Mel spectrogram predicted by Tacotron 2", loc="left")
    ax_mel.set_xlabel("Time (s)")
    ax_mel.set_ylabel("Mel band")

    image = ax_att.imshow(alignment.T, origin="lower", aspect="auto", cmap="Blues",
                          vmin=0, vmax=1, interpolation="none")
    ax_att.set_title("Attention alignment (diagonal = reads text in order)", loc="left")
    ax_att.set_xlabel("Decoder step (mel frame)")
    ax_att.set_ylabel("Input character")
    fig.colorbar(image, ax=ax_att, label="Attention weight")

    fig.suptitle(f'"{text}"', fontsize=10, wrap=True)
    for ax in (ax_mel, ax_att):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    return fig


def plot(mel, alignment, text, out_path):
    fig = make_figure(mel, alignment, text)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot Tacotron 2 attention alignments")
    parser.add_argument("--text", action="append", help="sentence to plot (repeatable)")
    parser.add_argument("--out-dir", default="samples/alignments")
    args = parser.parse_args()

    sentences = args.text or DEFAULT_SENTENCES
    for i, text in enumerate(sentences, start=1):
        mel, _, alignment = text_to_mel(text)
        mel, alignment = mel[0], alignment[0]
        out_path = Path(args.out_dir) / f"alignment_{i:02d}.png"
        plot(mel.numpy(), alignment.numpy(), text, out_path)

        stats = alignment_stats(alignment)
        print(f"[{i}] {out_path}  frames={alignment.shape[0]} chars={alignment.shape[1]}  "
              f"monotonic={stats['monotonic']:.1%}  coverage={stats['coverage']:.1%}  "
              f"focus={stats['focus']:.2f}")


if __name__ == "__main__":
    main()
