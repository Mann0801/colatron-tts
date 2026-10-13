"""Figures and numbers about Colatron for the README, slides and write-up.

    python src/report_figures.py --runs data/colatron_r3 data/colatron_colab

Writes
    samples/colatron/training_progress.png   attention at three training steps + learning curves
    samples/colatron/alignment_02.png        Colatron's mel and attention on sentence 02
    samples/colatron/stats.json              alignment stats, training length, model size
"""
import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from plot_alignment import DEFAULT_SENTENCES, make_figure  # noqa: E402
from src.models.colatron import ColatronSynth, alignment_stats  # noqa: E402

OUT = ROOT / "samples" / "colatron"


def read_runs(runs):
    """Merge the logs of a training that was resumed in another place (laptop, then Colab).

    Each later run starts with "Resumed at step N"; earlier runs are cut at that step.
    Returns val steps, val losses, monitor plots {step: png} and training hours. Hours are
    summed per run, because each machine logs in its own time zone."""
    val, monitors, hours = {}, {}, 0.0
    parsed = []
    for run in runs:
        lines = (run / "train.log").read_text().splitlines()
        resumed = [int(m.group(1)) for m in (re.search(r"Resumed at step (\d+)", l) for l in lines) if m]
        parsed.append((run, lines, resumed[0] if resumed else 0))
    for k, (run, lines, _) in enumerate(parsed):
        cut = parsed[k + 1][2] if k + 1 < len(parsed) else float("inf")
        last_step, prev = 0, None
        for line in lines:
            m = re.match(r"\[(\d\d:\d\d:\d\d)\] step\s+(\d+)", line)
            if m:
                last_step = int(m.group(2))
                if last_step > cut:
                    break
                cur = datetime.strptime(m.group(1), "%H:%M:%S")
                if prev is not None:
                    d = (cur - prev).total_seconds()
                    hours += (d if d >= 0 else d + 86400) / 3600
                prev = cur
            if "Resumed at step" in line:
                prev = None
            m = re.search(r"val mel loss ([\d.]+)", line)
            if m:
                val[last_step] = float(m.group(1))
        for png in (run / "monitor").glob("step_*.png"):
            step = int(png.stem.split("_")[1])
            if step <= cut:
                monitors[step] = png
    steps = sorted(val)
    return steps, [val[s] for s in steps], dict(sorted(monitors.items())), hours


def recogniser_curve(monitors):
    """Character error rate of the speech recogniser on each monitor clip (an unseen sentence)."""
    import soundfile as sf
    from eval.intelligibility import error_rates, normalise, transcribe
    from src.models.colatron import MONITOR_TEXT

    steps, cers = [], []
    for step, png in monitors.items():
        wav_path = png.with_suffix(".wav")
        if wav_path.exists():
            wav, sr = sf.read(wav_path, dtype="float32")
            steps.append(step)
            cers.append(100 * error_rates(normalise(MONITOR_TEXT), transcribe(wav, sr))[1])
    return steps, cers


def progress_figure(val_steps, val_losses, monitors, out_path):
    pngs = list(monitors.values())
    picks = [pngs[0], pngs[len(pngs) // 2], pngs[-1]] if len(pngs) >= 3 else pngs
    cer_steps, cers = recogniser_curve(monitors)

    fig = plt.figure(figsize=(12, 6.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 6, height_ratios=[1, 1])
    for k, path in enumerate(picks):
        img = plt.imread(path)
        ax = fig.add_subplot(grid[0, 2 * k:2 * k + 2])
        ax.imshow(img[int(img.shape[0] * 0.53):], aspect="auto")   # bottom half of the monitor plot: attention
        ax.set_title(f"Attention after {int(path.stem.split('_')[1]):,} steps", loc="left", fontsize=11)
        ax.axis("off")

    ax = fig.add_subplot(grid[1, :3])
    ax.plot(val_steps, val_losses, color="#5B3FD9", marker="o", markersize=3)
    ax.set_title("Validation mel loss (lower is better)", loc="left")
    ax.set_xlabel("Training step")
    ax2 = fig.add_subplot(grid[1, 3:])
    if cers:
        ax2.plot(cer_steps, cers, color="#FF6B57", marker="o", markersize=3)
        ax2.set_ylim(0, 105)
    ax2.set_title("Recogniser character error, unseen sentence (%)", loc="left")
    ax2.set_xlabel("Training step")
    for a in (ax, ax2):
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=[str(ROOT / "data" / "colatron_r3")],
                    help="training run folders in order (e.g. laptop then Colab)")
    ap.add_argument("--ckpt", default=str(ROOT / "data" / "colatron" / "colatron.pt"))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    val_steps, val_losses, monitors, hours = read_runs([Path(r) for r in args.runs])
    progress_figure(val_steps, val_losses, monitors, OUT / "training_progress.png")

    synth = ColatronSynth(args.ckpt)
    stats = []
    for i, text in enumerate(DEFAULT_SENTENCES, start=1):
        mel, align = synth.mel(text)
        st = alignment_stats(torch.from_numpy(align))
        stats.append(st)
        print(f"[{i}] monotonic {st['monotonic']:.1%}  coverage {st['coverage']:.1%}  focus {st['focus']:.2f}")
        if i == 2:
            fig = make_figure(mel, align, text, model_name="Colatron (ours)")
            fig.savefig(OUT / "alignment_02.png", dpi=120)
            plt.close(fig)

    summary = {
        "checkpoint_step": synth.step,
        "train_steps": val_steps[-1] if val_steps else None,
        "train_hours": round(hours, 1),
        "parameters": sum(p.numel() for p in synth.model.parameters()),
        "reduction_factor": synth.model.r,
        "monotonic_min": min(s["monotonic"] for s in stats),
        "monotonic_max": max(s["monotonic"] for s in stats),
        "coverage_min": min(s["coverage"] for s in stats),
        "coverage_max": max(s["coverage"] for s in stats),
        "best_val_loss": min(val_losses) if val_losses else None,
    }
    (OUT / "stats.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Saved figures and stats to {OUT}")


if __name__ == "__main__":
    main()
