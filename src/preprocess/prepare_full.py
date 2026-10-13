"""Prepare ALL of LJ Speech (13,100 clips) for training Colatron.

    python -m src.preprocess.prepare_full                  # uses data/LJSpeech-1.1.tar.bz2
    python -m src.preprocess.prepare_full --archive PATH

Uses the same text cleaning, fixed vocabulary and mel settings as prepare.py.
The test split is the SAME 40 test clips as the 404-clip subset (and the subset's
40 validation clips stay in validation), so scores stay comparable with the baseline
and none of those clips is ever trained on.

Writes data/full/
    wavs/<id>.wav          extracted audio
    mels.npy               all log-mels concatenated along time, float16, (total_frames, 80)
    index.csv              id|start|n_frames|clean_text   (rows of mels.npy for each clip)
    metadata.csv           id|raw_text   (original transcripts, used by eval/objective.py)
    splits.json            train / val / test ids
"""
import argparse
import json
import random
import sys
import tarfile
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.preprocess.audio_features import load_wav, wav_to_mel  # noqa: E402
from src.preprocess.common import DATA_DIR, audio_config  # noqa: E402
from src.preprocess.download import LJ_TOTAL_CLIPS  # noqa: E402
from src.preprocess.text_cleaning import clean_text  # noqa: E402

FULL_DIR = DATA_DIR / "full"
WAV_DIR = FULL_DIR / "wavs"
SUBSET_EVERY = round(LJ_TOTAL_CLIPS / 400)  # download.py's default subset (404 clips)
EXTRA_VAL = 60                              # validation clips added from outside the subset


def extract(archive):
    """Extract every wav and return {id: raw_text}."""
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    metadata, n = None, 0
    with tarfile.open(archive, "r|bz2") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = Path(member.name).name
            if name == "metadata.csv":
                metadata = tar.extractfile(member).read().decode("utf-8")
            elif name.endswith(".wav"):
                out = WAV_DIR / name
                if not out.exists():
                    out.write_bytes(tar.extractfile(member).read())
                n += 1
                if n % 1000 == 0:
                    print(f"  extracted {n} clips", flush=True)
    if metadata is None:
        raise RuntimeError("metadata.csv was not found inside the archive")
    texts = {}
    for line in metadata.splitlines():
        parts = line.split("|")
        if len(parts) >= 2:
            texts[parts[0]] = parts[1].strip()
    return texts


def subset_split(ids, seed=0, frac=0.1):
    """Re-create prepare.py's split of the 404-clip subset (same hash, same shuffle)."""
    sub = sorted(i for i in ids if zlib.crc32(i.encode("utf-8")) % SUBSET_EVERY == 0)
    random.Random(seed).shuffle(sub)
    n = max(1, round(len(sub) * frac))
    return sorted(sub[:n]), sorted(sub[n:2 * n])


def compute(item):
    clip_id, text = item
    y = load_wav(WAV_DIR / f"{clip_id}.wav", audio_config()["sr"])
    return clip_id, text, wav_to_mel(y).T.astype(np.float16)  # (frames, 80)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", type=Path, default=DATA_DIR / "LJSpeech-1.1.tar.bz2")
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()
    if not args.archive.exists():
        sys.exit(f"Archive not found: {args.archive}")

    print("Extracting all clips ...")
    texts = extract(args.archive)
    with open(FULL_DIR / "metadata.csv", "w", encoding="utf-8") as f:
        f.write("id|raw_text\n")
        f.writelines(f"{i}|{t}\n" for i, t in sorted(texts.items()))
    items = sorted((i, clean_text(t)) for i, t in texts.items())
    items = [(i, t) for i, t in items if t and (WAV_DIR / f"{i}.wav").exists()]

    print(f"Computing mels for {len(items)} clips with {args.workers} workers ...")
    with Pool(args.workers) as pool:
        results = []
        for k, r in enumerate(pool.imap(compute, items, chunksize=32), start=1):
            results.append(r)
            if k % 1000 == 0:
                print(f"  {k}/{len(items)}", flush=True)

    start, rows = 0, []
    for clip_id, text, mel in results:
        rows.append((clip_id, start, len(mel), text))
        start += len(mel)
    np.save(FULL_DIR / "mels.npy", np.concatenate([m for _, _, m in results]))
    with open(FULL_DIR / "index.csv", "w", encoding="utf-8") as f:
        f.write("id|start|n_frames|clean_text\n")
        for r in rows:
            f.write("|".join(map(str, r)) + "\n")

    ids = [r[0] for r in rows]
    test, val = subset_split(ids)
    rest = sorted(set(ids) - set(test) - set(val))
    random.Random(1).shuffle(rest)
    val = sorted(val + rest[:EXTRA_VAL])
    train = sorted(rest[EXTRA_VAL:])
    with open(FULL_DIR / "splits.json", "w", encoding="utf-8") as f:
        json.dump({"test": test, "val": val, "train": train}, f, indent=2)

    hours = start * audio_config()["hop"] / audio_config()["sr"] / 3600
    print(f"Done: {len(rows)} clips ({hours:.1f} h) -> train {len(train)}, val {len(val)}, test {len(test)}")


if __name__ == "__main__":
    main()
