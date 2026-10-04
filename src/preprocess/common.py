"""Shared helpers for preprocessing, the baseline model and evaluation.

Nothing in here imports torch, so the preprocessing scripts stay lightweight.
"""
import csv
import importlib
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_WAVS = RAW_DIR / "wavs"
PROC_DIR = DATA_DIR / "processed"
MEL_DIR = PROC_DIR / "mels"
BASELINE_CKPT = DATA_DIR / "baseline" / "baseline.pt"


def import_first(*names):
    """Import the first module in `names` that exists (the repo layout may vary)."""
    for name in names:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as e:
            missing = e.name or ""
            # Only swallow "this candidate does not exist"; re-raise real dependency errors.
            if not any(n == missing or n.startswith(missing + ".") for n in names):
                raise
    raise ImportError(f"Could not import any of: {', '.join(names)}")


# --------------------------------------------------------------------------- audio config
# The settings the team agreed on (src/audio_config.py). Used as fallback and as a cross-check.
SPEC = {
    "sr": 22050, "n_fft": 1024, "hop": 256, "win": 1024,
    "n_mels": 80, "fmin": 0.0, "fmax": 8000.0, "power": 1.0,
}
_ALIASES = {
    "sr": ["sample_rate", "sampling_rate", "sr"],
    "n_fft": ["n_fft", "nfft", "fft_size"],
    "hop": ["hop_length", "hop_size", "hop"],
    "win": ["win_length", "win_size", "win"],
    "n_mels": ["n_mels", "num_mels", "n_mel", "mel_bands"],
    "fmin": ["fmin", "f_min", "mel_fmin"],
    "fmax": ["fmax", "f_max", "mel_fmax"],
    "power": ["power"],
}
_CONTAINERS = ("cfg", "config", "audio", "audio_config", "audioconfig", "default")


def _find(objs, names):
    for obj in objs:
        for name in names:
            for key in (name, name.lower(), name.upper()):
                if isinstance(obj, dict):
                    if key not in obj:
                        continue
                    val = obj[key]
                else:
                    if not hasattr(obj, key):
                        continue
                    val = getattr(obj, key)
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    return val
    return None


@lru_cache(maxsize=1)
def audio_config():
    """Return the mel settings, read from src/audio_config.py when possible."""
    cfg = dict(SPEC)
    try:
        mod = import_first("audio_config", "src.audio_config")
    except ImportError:
        mod = None
    if mod is not None:
        objs = [mod]
        for attr in dir(mod):
            if attr.lower() in _CONTAINERS:
                objs.append(getattr(mod, attr))
        for key, names in _ALIASES.items():
            val = _find(objs, names)
            if val is not None:
                cfg[key] = val
    for k in ("sr", "n_fft", "hop", "win", "n_mels"):
        cfg[k] = int(cfg[k])
    for k in ("fmin", "fmax", "power"):
        cfg[k] = float(cfg[k])
    diffs = {k: (cfg[k], SPEC[k]) for k in SPEC if abs(cfg[k] - SPEC[k]) > 1e-9}
    if diffs:
        print(f"[warning] audio_config differs from the agreed settings: {diffs}", file=sys.stderr)
    return cfg


# --------------------------------------------------------------------------- audio helpers
def to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def save_wav(path, wav, sr):
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = np.asarray(wav, dtype=np.float32)
    peak = float(np.abs(wav).max()) if wav.size else 0.0
    if peak > 0.99:
        wav = wav * (0.99 / peak)
    sf.write(str(path), wav, int(sr), subtype="PCM_16")


def resample_to(wav, sr_in, sr_out):
    if int(sr_in) == int(sr_out):
        return wav
    import librosa

    return librosa.resample(wav, orig_sr=int(sr_in), target_sr=int(sr_out))


def mel_to_audio(mel):
    """Run the team's Griffin-Lim vocoder (src/vocoder/griffinlim.py: mel_to_wav).

    `mel` is a (n_mels, frames) natural-log mel. Returns (wav float32 1-D, sample_rate).
    """
    cfg = audio_config()
    mod = import_first("vocoder.griffinlim", "src.vocoder.griffinlim")
    fn = getattr(mod, "mel_to_wav")
    mel = np.ascontiguousarray(mel, dtype=np.float32)

    candidates = [lambda: mel, lambda: mel[None]]
    try:
        import torch

        candidates += [lambda: torch.from_numpy(mel), lambda: torch.from_numpy(mel)[None]]
    except Exception:  # torch missing or broken (e.g. a DLL error): numpy inputs are still tried
        pass

    errors = []
    for make in candidates:
        try:
            out = fn(make())
        except Exception as e:  # try the next input format
            errors.append(f"{type(e).__name__}: {e}")
            continue
        sr = cfg["sr"]
        if isinstance(out, (tuple, list)):
            if len(out) > 1 and isinstance(out[1], (int, np.integer)):
                sr = int(out[1])
            out = out[0]
        wav = to_numpy(out).astype(np.float32).squeeze()
        if wav.ndim == 1 and wav.size > cfg["hop"] * 4:
            return wav, sr
        errors.append(f"unexpected output shape {wav.shape}")
    raise RuntimeError("mel_to_wav failed for every input format tried: " + " | ".join(errors))


# --------------------------------------------------------------------------- data files
def read_metadata():
    """id -> {"raw": ..., "clean": ...} from data/processed/metadata.csv."""
    path = PROC_DIR / "metadata.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m src.preprocess.prepare")
    meta = {}
    with open(path, encoding="utf-8") as f:
        next(f, None)  # header
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) >= 3:
                meta[parts[0]] = {"raw": parts[1], "clean": parts[2]}
    return meta


def read_splits():
    path = PROC_DIR / "splits.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m src.preprocess.prepare")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_sentences(path):
    """Read samples/sentences.csv -> list of (name, text). Copes with several layouts."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.reader(f) if any(c.strip() for c in r)]
    if not rows:
        raise ValueError(f"{path} is empty")

    header = [c.strip().lower() for c in rows[0]]
    text_keys = ("text", "sentence", "transcript", "utterance")
    id_names = ("id", "idx", "index", "name", "file", "filename")

    text_col = next(
        (i for i, h in enumerate(header) if len(h) < 25 and any(k in h for k in text_keys)), None
    )
    id_col = next((header.index(n) for n in id_names if n in header), None)

    if text_col is not None:
        body = rows[1:]
        ncols = len(header)
    else:
        body, id_col = rows, None
        lengths = [len(r) for r in rows]
        ncols = max(set(lengths), key=lengths.count)  # most common row width
        if ncols == 1:
            text_col = 0
        else:
            avg = [sum(len(r[c]) for r in rows if c < len(r)) / len(rows) for c in range(ncols)]
            text_col = max(range(ncols), key=lambda c: avg[c])

    out = []
    for i, r in enumerate(body, start=1):
        if text_col >= len(r):
            continue
        # if the text is the last column and was not quoted, commas inside it split it up: rejoin
        text = (",".join(r[text_col:]) if text_col == ncols - 1 else r[text_col]).strip()
        if not text:
            continue
        name = ""
        if id_col is not None and id_col < len(r):
            name = r[id_col].strip()
        name = re.sub(r"\.wav$", "", name, flags=re.IGNORECASE)
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", name) or f"{i:02d}"
        out.append((name, text))
    return out
