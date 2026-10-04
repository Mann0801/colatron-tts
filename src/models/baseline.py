"""Baseline text -> mel model, trained from scratch on a few hundred clips.

It mimics the paper's weak baselines: characters go through a small convolutional
encoder, and the result is stretched to the target length by plain interpolation,
with NO learned alignment and NO attention. It is SUPPOSED to sound bad.

    python -m src.models.baseline train                 # trains, saves data/baseline/baseline.pt
    python -m src.models.baseline generate              # synthesises samples/sentences.csv -> samples/baseline/

(data/ is gitignored, so the checkpoint is never committed.)
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.preprocess.common import (  # noqa: E402
    BASELINE_CKPT, MEL_DIR, ROOT, audio_config, mel_to_audio, read_metadata, read_sentences,
    read_splits, save_wav,
)
from src.preprocess.text_cleaning import VOCAB_SIZE, clean_text, encode  # noqa: E402


# ----------------------------------------------------------------------------- model
class BaselineTTS(nn.Module):
    """chars -> embedding -> 3 conv layers -> stretch to n_frames -> MLP -> mel frames."""

    def __init__(self, vocab_size=VOCAB_SIZE, n_mels=80, emb=128, hidden=256, kernel=5, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(emb if i == 0 else hidden, hidden, kernel, padding=kernel // 2) for i in range(3)]
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_mels))

    def forward(self, tokens, lengths, n_frames):
        """tokens (B, L) long; lengths and n_frames are lists of ints. Returns list of (T_b, n_mels)."""
        x = self.embed(tokens).transpose(1, 2)  # (B, emb, L)
        for conv in self.convs:
            x = self.drop(F.relu(conv(x)))      # (B, hidden, L)
        outputs = []
        for b in range(tokens.size(0)):
            h = x[b:b + 1, :, :lengths[b]]                                     # (1, hidden, L_b)
            h = F.interpolate(h, size=int(n_frames[b]), mode="linear", align_corners=False)
            outputs.append(self.head(h.transpose(1, 2))[0])                    # (T_b, n_mels)
        return outputs


# ----------------------------------------------------------------------------- data
def load_split(split):
    meta = read_metadata()
    items = []
    for clip_id in read_splits()[split]:
        tokens = np.array(encode(meta[clip_id]["clean"]), dtype=np.int64)
        mel = np.load(MEL_DIR / f"{clip_id}.npy")
        items.append((clip_id, tokens, mel))
    return items


def make_batch(items, mean, std):
    _, toks, mels = zip(*items)
    lengths = [len(t) for t in toks]
    tokens = torch.zeros(len(toks), max(lengths), dtype=torch.long)
    for i, t in enumerate(toks):
        tokens[i, :len(t)] = torch.from_numpy(t)
    n_frames = [m.shape[1] for m in mels]
    targets = [torch.from_numpy(((m - mean[:, None]) / std[:, None]).T.astype(np.float32)) for m in mels]
    return tokens, lengths, n_frames, targets


def batch_loss(model, items, mean, std, device):
    tokens, lengths, n_frames, targets = make_batch(items, mean, std)
    preds = model(tokens.to(device), lengths, n_frames)
    losses = [F.l1_loss(p, t.to(device)) for p, t in zip(preds, targets)]
    return sum(losses) / len(losses)


# ----------------------------------------------------------------------------- training
def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_items, val_items = load_split("train"), load_split("val")
    if not train_items:
        sys.exit("No training clips. Run: python -m src.preprocess.prepare")
    print(f"Training on {len(train_items)} clips, validating on {len(val_items)} ({device})")

    all_mels = np.concatenate([m for _, _, m in train_items], axis=1)
    mean = all_mels.mean(axis=1).astype(np.float32)
    std = (all_mels.std(axis=1) + 1e-5).astype(np.float32)
    frames_per_char = sum(m.shape[1] for _, _, m in train_items) / sum(len(t) for _, t, _ in train_items)

    n_mels = train_items[0][2].shape[0]
    model = BaselineTTS(n_mels=n_mels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    BASELINE_CKPT.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_items)
        total = 0.0
        for i in range(0, len(train_items), args.batch_size):
            loss = batch_loss(model, train_items[i:i + args.batch_size], mean, std, device)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(train_items[i:i + args.batch_size])
        train_loss = total / len(train_items)

        model.eval()
        with torch.no_grad():
            vtotal = 0.0
            for i in range(0, len(val_items), args.batch_size):
                chunk = val_items[i:i + args.batch_size]
                vtotal += batch_loss(model, chunk, mean, std, device).item() * len(chunk)
            val_loss = vtotal / max(1, len(val_items))

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model": {k: v.cpu() for k, v in model.state_dict().items()},
                "mean": torch.from_numpy(mean),
                "std": torch.from_numpy(std),
                "frames_per_char": float(frames_per_char),
                "vocab_size": int(VOCAB_SIZE),
                "n_mels": int(n_mels),
            }, BASELINE_CKPT)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:3d}  train L1 {train_loss:.4f}  val L1 {val_loss:.4f}  (best {best_val:.4f})", flush=True)

    print(f"Best checkpoint saved to {BASELINE_CKPT}")


# ----------------------------------------------------------------------------- inference
class BaselineSynth:
    """Loads the trained baseline and turns text into a log-mel (n_mels, frames)."""

    def __init__(self, ckpt_path=BASELINE_CKPT):
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"{ckpt_path} not found. Train first: python -m src.models.baseline train")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.model = BaselineTTS(vocab_size=ckpt["vocab_size"], n_mels=ckpt["n_mels"])
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.mean, self.std = ckpt["mean"], ckpt["std"]
        self.frames_per_char = ckpt["frames_per_char"]

    @torch.no_grad()
    def mel(self, text):
        ids = encode(clean_text(text))
        if not ids:
            raise ValueError(f"Nothing left to synthesise after cleaning: {text!r}")
        n_frames = max(8, int(round(len(ids) * self.frames_per_char)))
        tokens = torch.tensor([ids], dtype=torch.long)
        pred = self.model(tokens, [len(ids)], [n_frames])[0]                  # (T, n_mels)
        mel = pred.t() * self.std[:, None] + self.mean[:, None]               # (n_mels, T)
        return mel.numpy().astype(np.float32)

    def wav(self, text):
        return mel_to_audio(self.mel(text))                                   # (wav, sr)


def generate(args):
    sentences = read_sentences(args.sentences)
    synth = BaselineSynth()
    out_dir = Path(args.out)
    for name, text in sentences:
        wav, sr = synth.wav(text)
        save_wav(out_dir / f"{name}.wav", wav, sr)
        print(f"  {name}.wav  <- {text[:70]}")
    print(f"Wrote {len(sentences)} files to {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train the baseline on the processed subset")
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(func=train)

    g = sub.add_parser("generate", help="synthesise samples/sentences.csv with the baseline")
    g.add_argument("--sentences", default=str(ROOT / "samples" / "sentences.csv"))
    g.add_argument("--out", default=str(ROOT / "samples" / "baseline"))
    g.set_defaults(func=generate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
