"""Intelligibility: can a speech recogniser understand the generated audio?

A pretrained wav2vec 2.0 recogniser (torchaudio WAV2VEC2_ASR_BASE_960H, trained on
LibriSpeech, never on LJ Speech) transcribes each clip, and we compare its transcript with
the sentence we asked for:
    WER = word error rate       (share of words wrong, missing or extra; lower is better)
    CER = character error rate  (same, per character)
The real recordings give the floor (what the recogniser gets wrong on its own).

    python eval/intelligibility.py score                       # all systems, first 20 test clips
    python eval/intelligibility.py watch --run data/colatron_r5   # follow a training run

`watch` waits for each new checkpoint (latest.pt), synthesises 5 test sentences with
HiFi-GAN, logs WER/CER to <run>/intelligibility.csv and saves the monitor sentence as
<run>/monitor/step_NNNNNN_hifigan.wav.
"""
import argparse
import csv
import json
import re
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torchaudio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from src.preprocess.common import import_first, save_wav  # noqa: E402
from src.preprocess.text_cleaning import clean_text  # noqa: E402

FULL_DIR = ROOT / "data" / "full"
ASR = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H


# ----------------------------------------------------------------------------- recogniser
@lru_cache(maxsize=1)
def load_asr():
    return ASR.get_model().eval(), ASR.get_labels()


@torch.inference_mode()
def transcribe(wav, sr):
    model, labels = load_asr()
    x = torch.as_tensor(np.asarray(wav, dtype=np.float32)).reshape(1, -1)
    x = torchaudio.functional.resample(x, int(sr), ASR.sample_rate)
    emissions, _ = model(x)
    ids = torch.unique_consecutive(emissions[0].argmax(-1)).tolist()  # greedy CTC decoding
    text = "".join(labels[i] for i in ids if i != 0).replace("|", " ")
    return normalise(text)


def normalise(text):
    """Lowercase words only: numbers spelled out, punctuation removed."""
    text = clean_text(text)
    return " ".join(re.sub(r"[^a-z' ]", " ", text).split())


def edit_distance(ref, hyp):
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def error_rates(ref, hyp):
    """(WER, CER) of one transcript; ref and hyp already normalised."""
    wer = edit_distance(ref.split(), hyp.split()) / max(1, len(ref.split()))
    cer = edit_distance(ref, hyp) / max(1, len(ref))
    return wer, cer


# ----------------------------------------------------------------------------- data
def test_set(n):
    ids = json.loads((FULL_DIR / "splits.json").read_text())["test"][:n]
    with open(FULL_DIR / "metadata.csv", encoding="utf-8") as f:
        next(f)
        texts = dict(line.rstrip("\n").split("|", 1) for line in f)
    return [(i, texts[i]) for i in ids]


# ----------------------------------------------------------------------------- watch
def watch(args):
    from src.models.colatron import MONITOR_TEXT, ColatronSynth

    torch.set_num_threads(args.threads)
    hifigan = import_first("vocoder.hifigan", "src.vocoder.hifigan")
    run = Path(args.run)
    ckpt = run / "latest.pt"
    log_path = run / "intelligibility.csv"
    done = set()
    if log_path.exists():
        with open(log_path, encoding="utf-8") as f:
            done = {int(r["step"]) for r in csv.DictReader(f)}
    sentences = test_set(args.n)
    print(f"Watching {ckpt} (scoring {len(sentences)} test sentences per checkpoint)", flush=True)

    last_mtime = None
    while True:
        if ckpt.exists() and ckpt.stat().st_mtime != last_mtime:
            last_mtime = ckpt.stat().st_mtime
            synth = ColatronSynth(ckpt)
            if synth.step not in done:
                wers, cers = [], []
                for k, (clip_id, text) in enumerate(sentences):
                    mel, _ = synth.mel(text)
                    wav, sr = hifigan.mel_to_wav(torch.from_numpy(mel)[None])
                    hyp = transcribe(wav, sr)
                    w, c = error_rates(normalise(text), hyp)
                    wers.append(w)
                    cers.append(c)
                    if k == 0:
                        example = hyp
                mel, _ = synth.mel(MONITOR_TEXT)
                wav, sr = hifigan.mel_to_wav(torch.from_numpy(mel)[None])
                save_wav(run / "monitor" / f"step_{synth.step:06d}_hifigan.wav", wav, sr)
                heard = transcribe(wav, sr)

                new = not log_path.exists()
                with open(log_path, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    if new:
                        w.writerow(["step", "wer", "cer", "monitor_heard_as"])
                    w.writerow([synth.step, f"{np.mean(wers):.3f}", f"{np.mean(cers):.3f}", heard])
                done.add(synth.step)
                print(f"[{time.strftime('%H:%M:%S')}] step {synth.step}: WER {np.mean(wers):.0%}  "
                      f"CER {np.mean(cers):.0%}  | '{sentences[0][1][:50]}' heard as '{example[:60]}'"
                      f"  | monitor heard as '{heard}'", flush=True)
        time.sleep(60)


# ----------------------------------------------------------------------------- score
def score(args):
    from src.models.baseline import BaselineSynth
    from src.models.colatron import ColatronSynth

    infer = import_first("infer", "src.infer")
    gl = import_first("vocoder.griffinlim", "src.vocoder.griffinlim")
    hifi = import_first("vocoder.hifigan", "src.vocoder.hifigan")
    import soundfile as sf

    baseline = BaselineSynth()
    colatron = ColatronSynth(args.ckpt)

    def systems(clip_id, text):
        wav, sr = sf.read(FULL_DIR / "wavs" / f"{clip_id}.wav", dtype="float32")
        yield "real", wav, sr
        yield "baseline", *baseline.wav(text)
        mel, _ = colatron.mel(text)
        mel = torch.from_numpy(mel)[None]
        yield "colatron_griffinlim", *gl.mel_to_wav(mel)
        yield "colatron_hifigan", *hifi.mel_to_wav(mel)
        torch.manual_seed(0)
        mel, lengths, _ = infer.text_to_mel(text)
        yield "tacotron_griffinlim", *gl.mel_to_wav(mel, lengths)
        yield "hifigan", *hifi.mel_to_wav(mel, lengths)

    rows = []
    sentences = test_set(args.n)
    for k, (clip_id, text) in enumerate(sentences, start=1):
        ref = normalise(text)
        for system, wav, sr in systems(clip_id, text):
            hyp = transcribe(wav, sr)
            rows.append((clip_id, system, *error_rates(ref, hyp), hyp))
        print(f"  scored {k}/{len(sentences)}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "intelligibility_per_clip.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clip", "system", "wer", "cer", "transcript"])
        w.writerows(rows)
    names = {"real": "Real recording", "baseline": "Baseline (ours) + Griffin-Lim",
             "colatron_griffinlim": "Colatron (ours) + Griffin-Lim",
             "colatron_hifigan": "Colatron (ours) + HiFi-GAN",
             "tacotron_griffinlim": "Tacotron 2 + Griffin-Lim", "hifigan": "Tacotron 2 + HiFi-GAN"}
    with open(out / "intelligibility.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "wer", "cer", "n_clips"])
        print(f"\n{'System':<32}{'WER':>8}{'CER':>8}")
        for system, name in names.items():
            sel = [r for r in rows if r[1] == system]
            wer, cer = np.mean([r[2] for r in sel]), np.mean([r[3] for r in sel])
            w.writerow([system, f"{wer:.4f}", f"{cer:.4f}", len(sel)])
            print(f"{name:<32}{wer:>8.1%}{cer:>8.1%}")
    print(f"\nSaved to {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("watch", help="score each new checkpoint of a training run")
    w.add_argument("--run", required=True)
    w.add_argument("--n", type=int, default=5)
    w.add_argument("--threads", type=int, default=2)
    w.set_defaults(func=watch)
    s = sub.add_parser("score", help="WER/CER of every system on the test clips")
    s.add_argument("--n", type=int, default=20)
    s.add_argument("--ckpt", default=str(ROOT / "data" / "colatron" / "colatron.pt"))
    s.add_argument("--out", default=str(ROOT / "eval" / "results"))
    s.set_defaults(func=score)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
