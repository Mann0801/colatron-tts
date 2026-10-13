"""Colatron: our seq2seq text -> mel model with attention, trained from scratch on LJ Speech.

This is the model the CS229 paper tried to build (a seq2seq LSTM with attention), with the
fixes that make attention work: a fixed character vocabulary, all 13,100 clips,
location-sensitive attention, several mel frames per decoder step (reduction factor r),
and a guided attention loss that pushes the alignment towards the diagonal early on.

    python -m src.models.colatron train                 # trains on data/full/, checkpoints in data/colatron/
    python -m src.models.colatron train --resume        # continue after a stop
    python -m src.models.colatron generate              # samples/sentences.csv -> samples/colatron_*/

Architecture (Tacotron 2 style, smaller so it trains on a CPU):
    chars -> embedding 256 -> 3 x conv(256, k5) -> BiLSTM 2x128            (encoder)
    previous frame -> prenet 256-128 (dropout always on)
    attention LSTM 512 -> location-sensitive attention -> decoder LSTM 512
    -> linear to r mel frames + stop token -> post-net 5 x conv(256, k5) residual
"""
import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.preprocess.common import DATA_DIR, ROOT, mel_to_audio, read_sentences, save_wav  # noqa: E402
from src.preprocess.text_cleaning import VOCAB_SIZE, clean_text, encode  # noqa: E402

FULL_DIR = DATA_DIR / "full"
OUT_DIR = DATA_DIR / "colatron"
COLATRON_CKPT = OUT_DIR / "colatron.pt"      # best validation weights, used for inference
LATEST_CKPT = OUT_DIR / "latest.pt"          # full training state, used by --resume
MONITOR_TEXT = "The quick brown fox jumps over the lazy dog."


# ----------------------------------------------------------------------------- model
class Encoder(nn.Module):
    def __init__(self, vocab_size, dim=256, kernel=5, dropout=0.5):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(dim, dim, kernel, padding=kernel // 2), nn.BatchNorm1d(dim),
                          nn.ReLU(), nn.Dropout(dropout))
            for _ in range(3)
        ])
        self.lstm = nn.LSTM(dim, dim // 2, batch_first=True, bidirectional=True)

    def forward(self, tokens, lengths):
        x = self.embed(tokens).transpose(1, 2)            # (B, dim, L)
        for conv in self.convs:
            x = conv(x)
        x = nn.utils.rnn.pack_padded_sequence(x.transpose(1, 2), lengths.cpu(), batch_first=True,
                                              enforce_sorted=False)
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        return x                                           # (B, L, dim)


class LocationAttention(nn.Module):
    """Additive attention that also sees where it looked before (previous + cumulative weights),
    which makes it move forward through the text instead of jumping around."""

    def __init__(self, query_dim, memory_dim, attn_dim=128, filters=32, kernel=31):
        super().__init__()
        self.query = nn.Linear(query_dim, attn_dim, bias=False)
        self.memory = nn.Linear(memory_dim, attn_dim, bias=False)
        self.loc_conv = nn.Conv1d(2, filters, kernel, padding=kernel // 2, bias=False)
        self.loc_dense = nn.Linear(filters, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1)

    def forward(self, query, memory, processed_memory, prev_weights, mask):
        loc = self.loc_dense(self.loc_conv(prev_weights).transpose(1, 2))       # (B, L, attn)
        energy = self.v(torch.tanh(self.query(query)[:, None] + processed_memory + loc)).squeeze(-1)
        weights = torch.softmax(energy.masked_fill(mask, -1e9), dim=-1)          # (B, L)
        context = torch.bmm(weights[:, None], memory).squeeze(1)                # (B, memory_dim)
        return context, weights


class Decoder(nn.Module):
    def __init__(self, n_mels, r, enc_dim=256, prenet=(256, 128), rnn_dim=512, dropout=0.1):
        super().__init__()
        self.n_mels, self.r, self.rnn_dim, self.dropout = n_mels, r, rnn_dim, dropout
        self.prenet = nn.ModuleList([nn.Linear(i, o) for i, o in zip((n_mels,) + prenet[:-1], prenet)])
        self.attn_rnn = nn.LSTMCell(prenet[-1] + enc_dim, rnn_dim)
        self.attention = LocationAttention(rnn_dim, enc_dim)
        self.dec_rnn = nn.LSTMCell(rnn_dim + enc_dim, rnn_dim)
        self.proj = nn.Linear(rnn_dim + enc_dim, n_mels * r)
        self.stop = nn.Linear(rnn_dim + enc_dim, 1)

    def run_prenet(self, x):
        for layer in self.prenet:
            x = F.dropout(F.relu(layer(x)), 0.5, training=True)   # on at inference too (Tacotron 2)
        return x

    def init_state(self, memory):
        B, L, D = memory.shape
        z = memory.new_zeros
        return {"ah": z(B, self.rnn_dim), "ac": z(B, self.rnn_dim), "dh": z(B, self.rnn_dim),
                "dc": z(B, self.rnn_dim), "ctx": z(B, D), "w": z(B, L), "wcum": z(B, L)}

    def step(self, x, s, memory, processed, mask):
        s["ah"], s["ac"] = self.attn_rnn(torch.cat([x, s["ctx"]], -1), (s["ah"], s["ac"]))
        s["ah"] = F.dropout(s["ah"], self.dropout, self.training)
        prev = torch.stack([s["w"], s["wcum"]], dim=1)
        s["ctx"], s["w"] = self.attention(s["ah"], memory, processed, prev, mask)
        s["wcum"] = s["wcum"] + s["w"]
        s["dh"], s["dc"] = self.dec_rnn(torch.cat([s["ah"], s["ctx"]], -1), (s["dh"], s["dc"]))
        s["dh"] = F.dropout(s["dh"], self.dropout, self.training)
        out = torch.cat([s["dh"], s["ctx"]], -1)
        return self.proj(out), self.stop(out).squeeze(-1), s["w"]

    def forward(self, memory, mask, mels):
        """Teacher forcing. mels (B, T, n_mels) with T a multiple of r."""
        B, T, _ = mels.shape
        steps = T // self.r
        prev = torch.cat([mels.new_zeros(B, 1, self.n_mels), mels[:, self.r - 1:-1:self.r]], 1)
        prev = self.run_prenet(prev)                                     # (B, steps, prenet)
        processed = self.attention.memory(memory)
        s = self.init_state(memory)
        frames, stops, aligns = [], [], []
        for t in range(steps):
            f, st, w = self.step(prev[:, t], s, memory, processed, mask)
            frames.append(f)
            stops.append(st)
            aligns.append(w)
        mel = torch.stack(frames, 1).view(B, T, self.n_mels)
        return mel, torch.stack(stops, 1), torch.stack(aligns, 1)       # (B,T,M) (B,steps) (B,steps,L)

    @torch.no_grad()
    def infer(self, memory, max_steps, threshold=0.5, end_margin=3):
        """Free-running synthesis. The stop token is only obeyed once the attention has reached
        the last `end_margin` characters, so a premature stop cannot cut the sentence short."""
        mask = torch.zeros(memory.shape[:2], dtype=torch.bool, device=memory.device)
        last = memory.shape[1] - 1 - end_margin
        processed = self.attention.memory(memory)
        s = self.init_state(memory)
        x = memory.new_zeros(1, self.n_mels)
        frames, aligns = [], []
        for _ in range(max_steps):
            f, st, w = self.step(self.run_prenet(x), s, memory, processed, mask)
            frames.append(f.view(1, self.r, self.n_mels))
            aligns.append(w)
            x = frames[-1][:, -1]
            if torch.sigmoid(st).item() > threshold and w.argmax().item() >= last:
                break
        return torch.cat(frames, 1), torch.stack(aligns, 1)


class Postnet(nn.Module):
    def __init__(self, n_mels, dim=256, kernel=5, dropout=0.5):
        super().__init__()
        chans = [n_mels, dim, dim, dim, dim, n_mels]
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(i, o, kernel, padding=kernel // 2), nn.BatchNorm1d(o))
            for i, o in zip(chans[:-1], chans[1:])
        ])
        self.dropout = dropout

    def forward(self, mel):                                  # (B, T, n_mels)
        x = mel.transpose(1, 2)
        for k, conv in enumerate(self.convs):
            x = conv(x)
            if k < len(self.convs) - 1:
                x = torch.tanh(x)
            x = F.dropout(x, self.dropout, self.training)
        return mel + x.transpose(1, 2)


class Colatron(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, n_mels=80, r=3):
        super().__init__()
        self.r = r
        self.encoder = Encoder(vocab_size)
        self.decoder = Decoder(n_mels, r)
        self.postnet = Postnet(n_mels)

    def forward(self, tokens, lengths, mels):
        memory = self.encoder(tokens, lengths)
        mask = torch.arange(tokens.shape[1], device=tokens.device)[None] >= lengths[:, None]
        mel, stop, align = self.decoder(memory, mask, mels)
        return mel, self.postnet(mel), stop, align

    @torch.no_grad()
    def infer(self, tokens, max_frames=1200):
        memory = self.encoder(tokens, torch.tensor([tokens.shape[1]]))
        mel, align = self.decoder.infer(memory, max_frames // self.r)
        return self.postnet(mel), align                     # (1, T, n_mels), (1, steps, L)


# ----------------------------------------------------------------------------- losses
def guided_attention_loss(align, text_lens, steps, g=0.2):
    """Penalty for attention far from the diagonal (Tachibana et al. 2017, DC-TTS)."""
    B, S, L = align.shape
    ar = lambda k: torch.arange(k, device=align.device)
    n = ar(L)[None, None].float() / text_lens[:, None, None].float()
    t = ar(S)[None, :, None].float() / steps[:, None, None].float()
    w = 1 - torch.exp(-((n - t) ** 2) / (2 * g * g))
    valid = (ar(S)[None, :, None] < steps[:, None, None]) & (ar(L)[None, None] < text_lens[:, None, None])
    return (align * w * valid).sum() / valid[:, :, 0].sum()     # mean penalty per decoder step


def compute_loss(model, batch, ga_weight):
    tokens, text_lens, mels, n_frames = batch
    mel, post, stop, align = model(tokens, text_lens, mels)
    T = mels.shape[1]
    fmask = (torch.arange(T, device=mels.device)[None] < n_frames[:, None]).unsqueeze(-1).float()
    denom = fmask.sum() * mels.shape[2]
    mel_loss = sum(((p - mels).abs() * fmask).sum() / denom + (((p - mels) ** 2) * fmask).sum() / denom
                   for p in (mel, post))
    steps = torch.div(n_frames + model.r - 1, model.r, rounding_mode="floor")
    stop_target = (torch.arange(stop.shape[1], device=stop.device)[None] >= steps[:, None] - 1).float()
    stop_loss = F.binary_cross_entropy_with_logits(stop, stop_target)
    ga_loss = guided_attention_loss(align, text_lens, steps)
    total = mel_loss + stop_loss + ga_weight * ga_loss
    return total, {"mel": mel_loss.item(), "stop": stop_loss.item(), "guided": ga_loss.item()}


def alignment_stats(align):
    """align (steps, L). Same three numbers as src/plot_alignment.py."""
    weights, focused = align.max(dim=1)
    return {"monotonic": (focused[1:] >= focused[:-1]).float().mean().item(),
            "coverage": focused.unique().numel() / align.shape[1],
            "focus": weights.mean().item()}


# ----------------------------------------------------------------------------- data
class FullData:
    """All mels in RAM (float16), length-bucketed batches."""

    def __init__(self):
        if not (FULL_DIR / "index.csv").exists():
            sys.exit("data/full/ not found. Run: python -m src.preprocess.prepare_full")
        self.mels = np.load(FULL_DIR / "mels.npy", mmap_mode="r")
        self.mels = np.ascontiguousarray(self.mels)
        self.index = {}
        with open(FULL_DIR / "index.csv", encoding="utf-8") as f:
            next(f)
            for line in f:
                clip_id, start, n, text = line.rstrip("\n").split("|", 3)
                self.index[clip_id] = (int(start), int(n), np.array(encode(text), dtype=np.int64))
        self.splits = json.loads((FULL_DIR / "splits.json").read_text())

    def mel(self, clip_id):
        start, n, _ = self.index[clip_id]
        return self.mels[start:start + n].astype(np.float32)

    def stats(self):
        rows = np.concatenate([self.mel(i) for i in self.splits["train"][::10]])
        return rows.mean(0).astype(np.float32), (rows.std(0) + 1e-5).astype(np.float32)

    def batches(self, split, batch_size, shuffle, rng=None):
        ids = list(self.splits[split])
        if shuffle:
            # sort by length with a little noise, cut into batches, shuffle the batches
            ids.sort(key=lambda i: self.index[i][1] * rng.uniform(0.9, 1.1))
            groups = [ids[k:k + batch_size] for k in range(0, len(ids), batch_size)]
            rng.shuffle(groups)
            return groups
        ids.sort(key=lambda i: self.index[i][1])
        return [ids[k:k + batch_size] for k in range(0, len(ids), batch_size)]

    def collate(self, ids, mean, std, r):
        toks = [self.index[i][2] for i in ids]
        mels = [(self.mel(i) - mean) / std for i in ids]
        L = max(len(t) for t in toks)
        T = max(len(m) for m in mels)
        T = int(math.ceil(T / r) * r)
        tokens = torch.zeros(len(ids), L, dtype=torch.long)
        out = torch.zeros(len(ids), T, mels[0].shape[1])
        for k, (t, m) in enumerate(zip(toks, mels)):
            tokens[k, :len(t)] = torch.from_numpy(t)
            out[k, :len(m)] = torch.from_numpy(m)
        return (tokens, torch.tensor([len(t) for t in toks]), out,
                torch.tensor([len(m) for m in mels]))


# ----------------------------------------------------------------------------- training
def lr_at(step, base, decay_start, half_life, floor=1e-4):
    if step < decay_start:
        return base
    return max(floor, base * 0.5 ** ((step - decay_start) / half_life))


def save_monitor(model, mean, std, step, log, out_dir):
    """Free-running synthesis of one sentence: alignment plot + Griffin-Lim audio + stats."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    torch.manual_seed(0)
    device = next(model.parameters()).device
    mel, align = model.infer(torch.tensor([encode(clean_text(MONITOR_TEXT))], device=device))
    model.train()
    mel, align = mel.cpu(), align.cpu()
    mel = (mel[0].numpy() * std + mean).T                   # (80, T) natural log
    stats = alignment_stats(align[0])
    out = out_dir / "monitor"
    out.mkdir(parents=True, exist_ok=True)
    fig, (a, b) = plt.subplots(2, 1, figsize=(9, 6), constrained_layout=True)
    a.imshow(mel, origin="lower", aspect="auto", cmap="Purples")
    a.set_title(f"step {step}: predicted mel ({mel.shape[1]} frames)", loc="left")
    b.imshow(align[0].numpy().T, origin="lower", aspect="auto", cmap="Blues", vmin=0, vmax=1,
             interpolation="none")
    b.set_title(f"attention  monotonic {stats['monotonic']:.0%}  coverage {stats['coverage']:.0%}  "
                f"focus {stats['focus']:.2f}", loc="left")
    b.set_xlabel("decoder step")
    b.set_ylabel("character")
    fig.savefig(out / f"step_{step:06d}.png", dpi=90)
    plt.close(fig)
    wav, sr = mel_to_audio(mel)
    save_wav(out / f"step_{step:06d}.wav", wav, sr)
    log(f"  monitor: {mel.shape[1]} frames, monotonic {stats['monotonic']:.1%}, "
        f"coverage {stats['coverage']:.1%}, focus {stats['focus']:.2f}")


def train(args):
    torch.set_num_threads(args.threads)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt, latest_ckpt = out_dir / COLATRON_CKPT.name, out_dir / LATEST_CKPT.name
    log_file = open(out_dir / "train.log", "a", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    data = FullData()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Colatron(r=args.r).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    step, epoch, best_val = 0, 0, float("inf")
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    if args.resume and latest_ckpt.exists():
        ck = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, epoch, best_val = ck["step"], ck["epoch"], ck["best_val"]
        mean, std = ck["mean"].numpy(), ck["std"].numpy()
        rng.setstate(ck["rng"])
        log(f"Resumed at step {step}, epoch {epoch}")
    else:
        mean, std = data.stats()
    n_params = sum(p.numel() for p in model.parameters())
    log(f"Colatron: {n_params:,} parameters, r={args.r}, train {len(data.splits['train'])} clips, "
        f"val {len(data.splits['val'])}, batch {args.batch_size}, {args.threads} threads, {device}")

    def checkpoint(path, full):
        state = {"model": {k: v.cpu() for k, v in model.state_dict().items()}, "mean": torch.from_numpy(mean), "std": torch.from_numpy(std),
                 "config": {"r": args.r, "vocab_size": VOCAB_SIZE, "n_mels": 80}, "step": step}
        if full:
            state.update(opt=opt.state_dict(), epoch=epoch, best_val=best_val, rng=rng.getstate())
        tmp = path.with_suffix(".tmp")
        torch.save(state, tmp)
        tmp.replace(path)

    def validate():
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for ids in data.batches("val", args.batch_size, shuffle=False):
                batch = [x.to(device) for x in data.collate(ids, mean, std, args.r)]
                _, parts = compute_loss(model, batch, 0.0)
                total += parts["mel"] * len(ids)
                n += len(ids)
        model.train()
        return total / n

    model.train()
    t0 = time.time()
    while step < args.max_steps:
        epoch += 1
        for ids in data.batches("train", args.batch_size, shuffle=True, rng=rng):
            for g in opt.param_groups:
                g["lr"] = lr_at(step, args.lr, args.decay_start, args.half_life)
            batch = [x.to(device) for x in data.collate(ids, mean, std, args.r)]
            loss, parts = compute_loss(model, batch, args.guided_weight)
            opt.zero_grad()
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if torch.isfinite(gnorm):
                opt.step()
            step += 1

            if step % args.log_every == 0:
                rate = args.log_every / (time.time() - t0)
                t0 = time.time()
                log(f"step {step:6d} ep {epoch:3d}  mel {parts['mel']:.3f}  stop {parts['stop']:.3f}  "
                    f"guided {parts['guided']:.3f}  lr {opt.param_groups[0]['lr']:.1e}  "
                    f"{rate * 3600:.0f} steps/h")
            if step % args.eval_every == 0:
                val = validate()
                if val < best_val:
                    best_val = val
                    checkpoint(best_ckpt, full=False)
                log(f"  val mel loss {val:.4f} (best {best_val:.4f})")
                save_monitor(model, mean, std, step, log, out_dir)
                checkpoint(latest_ckpt, full=True)
                t0 = time.time()
            if step >= args.max_steps:
                break
    checkpoint(latest_ckpt, full=True)
    log(f"Finished at step {step}. Best weights: {best_ckpt}")


# ----------------------------------------------------------------------------- inference
class ColatronSynth:
    """Loads trained Colatron weights and turns text into a log-mel + attention alignment."""

    def __init__(self, ckpt_path=COLATRON_CKPT):
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"{ckpt_path} not found. Train first: python -m src.models.colatron train")
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.model = Colatron(vocab_size=ck["config"]["vocab_size"], n_mels=ck["config"]["n_mels"],
                              r=ck["config"]["r"])
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.mean, self.std = ck["mean"], ck["std"]
        self.step = ck.get("step")

    def mel(self, text, seed=0):
        """Returns (mel (80, T) natural log, alignment (decoder steps, chars))."""
        ids = encode(clean_text(text))
        if not ids:
            raise ValueError(f"Nothing left to synthesise after cleaning: {text!r}")
        torch.manual_seed(seed)                     # the prenet's dropout is random
        mel, align = self.model.infer(torch.tensor([ids]))
        mel = mel[0] * self.std + self.mean
        return mel.t().numpy().astype(np.float32), align[0].numpy()


def generate(args):
    from src.preprocess.common import import_first

    hifigan = import_first("vocoder.hifigan", "src.vocoder.hifigan")
    synth = ColatronSynth(args.ckpt)
    sentences = read_sentences(args.sentences)
    for name, text in sentences:
        mel, _ = synth.mel(text)
        save_wav(Path(args.out) / "colatron_griffinlim" / f"{name}.wav", *mel_to_audio(mel))
        wav, sr = hifigan.mel_to_wav(torch.from_numpy(mel)[None])
        save_wav(Path(args.out) / "colatron_hifigan" / f"{name}.wav", wav, sr)
        print(f"  {name}.wav  <- {text[:70]}")
    print(f"Wrote {len(sentences)} sentences to {args.out}/colatron_griffinlim and colatron_hifigan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train Colatron on data/full/")
    t.add_argument("--r", type=int, default=3, help="mel frames per decoder step")
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--decay-start", type=int, default=8000)
    t.add_argument("--half-life", type=int, default=8000)
    t.add_argument("--guided-weight", type=float, default=1.0)
    t.add_argument("--max-steps", type=int, default=100000)
    t.add_argument("--log-every", type=int, default=50)
    t.add_argument("--eval-every", type=int, default=500)
    t.add_argument("--threads", type=int, default=12)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--resume", action="store_true")
    t.add_argument("--out-dir", default=str(OUT_DIR), help="where checkpoints, log and monitor plots go")
    t.set_defaults(func=train)

    g = sub.add_parser("generate", help="synthesise samples/sentences.csv with Colatron")
    g.add_argument("--sentences", default=str(ROOT / "samples" / "sentences.csv"))
    g.add_argument("--out", default=str(ROOT / "samples"))
    g.add_argument("--ckpt", default=str(COLATRON_CKPT))
    g.set_defaults(func=generate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
