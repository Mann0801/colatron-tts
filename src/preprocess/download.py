"""Download LJ Speech and keep a small subset (~400 clips) in data/raw/.

    python -m src.preprocess.download                      # download (~2.6 GB) + extract subset
    python -m src.preprocess.download --target-clips 500   # keep about 500 clips
    python -m src.preprocess.download --archive path/to/LJSpeech-1.1.tar.bz2   # already downloaded

The subset is picked by hashing the file names, so everyone who runs this gets the
same clips. data/ is gitignored: never commit it.
"""
import argparse
import ssl
import sys
import tarfile
import urllib.error
import urllib.request
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.preprocess.common import DATA_DIR, RAW_DIR, RAW_WAVS  # noqa: E402

URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
LJ_TOTAL_CLIPS = 13100


def _ssl_context():
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def download(url, dest):
    """Download with resume support."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    try:
        resp = urllib.request.urlopen(req, context=_ssl_context(), timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416:  # range not satisfiable -> file is already complete
            print(f"Archive already complete: {dest}")
            return
        raise
    with resp:
        if resp.status == 206:
            mode, total = "ab", have + int(resp.headers.get("Content-Length", 0))
        else:
            mode, have, total = "wb", 0, int(resp.headers.get("Content-Length", 0))
        done, last_print = have, 0
        with open(dest, mode) as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if done - last_print >= 50 * (1 << 20):
                    last_print = done
                    pct = f" ({100 * done / total:.0f}%)" if total else ""
                    print(f"  downloaded {done / 1e6:.0f} MB{pct}", flush=True)
    print(f"Download finished: {dest}")


def extract_subset(archive, every):
    """Single streaming pass over the archive; keeps wavs whose name hash % every == 0."""
    RAW_WAVS.mkdir(parents=True, exist_ok=True)
    kept, metadata = [], None
    with tarfile.open(archive, "r|bz2") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = Path(member.name).name
            if name == "metadata.csv":
                metadata = tar.extractfile(member).read().decode("utf-8")
            elif name.endswith(".wav"):
                stem = Path(name).stem
                if zlib.crc32(stem.encode("utf-8")) % every == 0:
                    (RAW_WAVS / name).write_bytes(tar.extractfile(member).read())
                    kept.append(stem)
                    if len(kept) % 100 == 0:
                        print(f"  extracted {len(kept)} clips", flush=True)
    if metadata is None:
        raise RuntimeError("metadata.csv was not found inside the archive")
    if not kept:
        raise RuntimeError("no wav files were extracted; is this the LJ Speech archive?")

    keep = set(kept)
    lines = ["id|raw_text"]
    for line in metadata.splitlines():
        parts = line.split("|")
        if len(parts) >= 2 and parts[0] in keep:
            lines.append(f"{parts[0]}|{parts[1].strip()}")
    (RAW_DIR / "metadata.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines) - 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-clips", type=int, default=400, help="approximate number of clips to keep")
    ap.add_argument("--every", type=int, default=None, help="advanced: keep 1 in N clips (overrides --target-clips)")
    ap.add_argument("--archive", type=Path, default=None, help="use an already downloaded LJSpeech-1.1.tar.bz2")
    ap.add_argument("--url", default=URL)
    ap.add_argument("--delete-archive", action="store_true", help="remove the 2.6 GB archive afterwards")
    args = ap.parse_args()

    every = args.every or max(1, round(LJ_TOTAL_CLIPS / args.target_clips))
    archive = args.archive or (DATA_DIR / "LJSpeech-1.1.tar.bz2")

    if args.archive is None:
        print(f"Downloading LJ Speech (~2.6 GB) to {archive} ...")
        download(args.url, archive)
    elif not archive.exists():
        sys.exit(f"Archive not found: {archive}")

    print(f"Extracting about 1 in {every} clips ...")
    n = extract_subset(archive, every)
    print(f"Done. {n} clips + transcripts are in {RAW_DIR}")

    if args.delete_archive and args.archive is None:
        archive.unlink()
        print("Archive deleted.")


if __name__ == "__main__":
    main()
