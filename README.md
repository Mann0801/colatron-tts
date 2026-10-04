# Colatron: End-to-End Text-to-Speech

UE24CS352A Machine Learning mini-project, PES University (Team 25, Project 20).

**Team:** Mann Mehta (PES2UG24CS266), Marapareddy Deekshitha (PES2UG24CS268)

We revisit the Stanford CS229 (2018) project *"End-to-End Text to Speech Synthesis"*
(Wang, Yang, Li). That project trained SVR, a small neural network, and a seq2seq LSTM
on about 200 LJ Speech sentences and reached a best MOS of 2.5 / 5, with attention that
never aligned (repeated words). We rebuild the pipeline with a fixed text mapping, a
standard mel-spectrogram front end, a pretrained Tacotron 2 acoustic model, and a
choice of vocoders, and compare it against our own from-scratch baseline.

```
text ──► text cleaning + fixed character ids ──► Tacotron 2 ──► 80-band mel ──► vocoder ──► .wav
                                                                              (Griffin-Lim or HiFi-GAN)
```

## What is pretrained and what we wrote

| Component | Source |
|---|---|
| Tacotron 2 acoustic model (text → mel) | **Pretrained**: torchaudio `TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH`, trained on all of LJ Speech |
| HiFi-GAN neural vocoder (mel → waveform) | **Pretrained**: SpeechBrain `speechbrain/tts-hifigan-ljspeech` |
| Griffin-Lim vocoder | **Ours**: inverse mel + Griffin-Lim from torchaudio transforms, no learning |
| Shared audio settings, inference pipeline, CLI | **Ours** |
| Attention alignment plots and alignment statistics | **Ours** |
| Fixed-sentence sample generation (same mel through both vocoders) | **Ours** |
| Gradio demo app | **Ours** |
| Data download, text cleaning, mel extraction, splits | **Ours** *(in progress)* |
| Baseline model trained from scratch | **Ours** *(in progress)* |
| Evaluation: MOS sheet, mel distance, results table | **Ours** *(in progress)* |

We did not train or fine-tune Tacotron 2 or HiFi-GAN. Our trained model is the baseline.

## Setup (Fedora / Linux, CPU only)

Requires **Python 3.11** (PyTorch does not yet support 3.14). On Fedora: `sudo dnf install python3.11`.

```bash
git clone git@github.com:Mann0801/colatron-tts.git
cd colatron-tts
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Pretrained weights download automatically on first run (about 200 MB in total) to
`~/.cache/torch` and `checkpoints/`. Neither is committed to the repo.

## Usage

### Synthesize one sentence

```bash
python src/infer.py --text "Hello, this is our project." --vocoder hifigan --out outputs/hello.wav
python src/infer.py --text "Hello, this is our project." --vocoder griffinlim --out outputs/hello_gl.wav
```

Prints the mel and alignment shapes, the audio length and the generation time.

### Live demo

```bash
python app/app.py
```

Open http://127.0.0.1:7860, type a sentence, pick a vocoder and press **Speak**. The page
plays the audio and shows the predicted mel spectrogram and attention alignment.
Pre-generated audio in `samples/` is a fallback if the live demo cannot run.

### Regenerate the comparison samples

```bash
python src/generate_samples.py
```

Synthesizes the 10 fixed sentences in `samples/sentences.csv`. Each sentence becomes a mel
spectrogram once (fixed seed), and that same mel goes through both vocoders, so the only
difference between `samples/main_griffinlim/NN.wav` and `samples/main_hifigan/NN.wav` is
the vocoder.

### Attention alignment plots

```bash
python src/plot_alignment.py                      # 3 default sentences
python src/plot_alignment.py --text "Your sentence here."
```

Writes plots to `samples/alignments/` and prints three statistics per sentence:
**monotonic** (share of decoder steps that do not move backwards in the text),
**coverage** (share of characters that were ever the main focus) and **focus**
(mean peak attention weight).

![Attention alignment](samples/alignments/alignment_02.png)

### Data and preprocessing *(in progress)*

<!-- Deekshitha: commands for downloading LJ Speech, preprocessing and splitting -->

### Baseline model *(in progress)*

<!-- Deekshitha: commands for training the baseline and generating samples/baseline/ -->

### Evaluation *(in progress)*

<!-- Deekshitha: commands for the MOS sheet and the mel-distance metric -->

## Audio settings

All parts of the project share `src/audio_config.py`, which matches the pretrained
Tacotron 2 and HiFi-GAN:

| Setting | Value |
|---|---|
| Sample rate | 22,050 Hz |
| FFT size / window / hop | 1024 / 1024 / 256 samples |
| Mel bands | 80, slaney scale and norm, 0 to 8000 Hz |
| Stored as | natural log of magnitude, `log(clamp(mel, 1e-5))` |

## Results so far

Measured on a 24-core laptop CPU, no GPU.

| Vocoder | Type | Time for 3 to 7 s of audio | Notes |
|---|---|---|---|
| WaveRNN (tried, removed) | autoregressive neural | ~49 s for 3.3 s | too slow for a live demo |
| Griffin-Lim | signal processing | 0.06 to 0.10 s | intelligible, metallic |
| HiFi-GAN | parallel neural (GAN) | 0.43 to 1.04 s | natural; about 7× faster than real time |

Tacotron 2 attention on 3 test sentences: **95 to 97% monotonic**, 84 to 88% character
coverage. A clean diagonal means the model reads the text in order, which the original
project's seq2seq model never achieved.

### MOS (to be filled in after the listening test)

| System | MOS (1 to 5) |
|---|---|
| CS229 2018: SVR | 1.0 |
| CS229 2018: simple NN | 1.7 |
| CS229 2018: seq2seq + attention | 2.5 |
| Tacotron (reported in original paper) | 3.82 |
| **Ours: baseline (trained from scratch)** | *TBD* |
| **Ours: Tacotron 2 + Griffin-Lim** | *TBD* |
| **Ours: Tacotron 2 + HiFi-GAN** | *TBD* |
| Real recording (LJ Speech) | *TBD* |

## Repository structure

```
src/
  audio_config.py        shared mel settings
  infer.py               CLI: text -> wav
  generate_samples.py    fixed sentences through both vocoders
  plot_alignment.py      mel + attention plots and alignment stats
  models/
    tacotron.py          pretrained Tacotron 2 wrapper
    baseline.py          our baseline (in progress)
  vocoder/
    griffinlim.py        Griffin-Lim vocoder
    hifigan.py           pretrained HiFi-GAN wrapper
  preprocess/            data download, text cleaning, mel extraction (in progress)
app/app.py               Gradio demo
eval/                    MOS sheet and objective metrics (in progress)
samples/                 committed demo audio and plots
report/                  write-up notes
data/, checkpoints/, outputs/   gitignored
```

## Who did what

| Member | Parts | Files |
|---|---|---|
| Mann Mehta | Pipeline setup, pretrained model integration, vocoders, alignment analysis, samples, inference CLI, demo, README | `src/models/tacotron.py`, `src/vocoder/`, `src/infer.py`, `src/plot_alignment.py`, `src/generate_samples.py`, `src/audio_config.py`, `app/`, `README.md` |
| Marapareddy Deekshitha | Data, preprocessing, baseline model, evaluation | `src/preprocess/`, `src/models/baseline.py`, `eval/` |

## Limitations

- Tacotron 2 and HiFi-GAN are pretrained on the full LJ Speech dataset, so LJ Speech
  test sentences may have been seen during their training. Mel-distance scores on those
  clips favour the pretrained system.
- Tacotron 2 keeps dropout on in its prenet at inference, so repeated runs differ slightly.
  `generate_samples.py` fixes a seed for reproducibility.
- Single speaker (LJ Speech), English only.

## References

1. X. Wang, Y. Yang, Y. Li. *End-to-End Text to Speech Synthesis.* CS229 project report, Stanford, 2018.
2. J. Shen et al. *Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions* (Tacotron 2). ICASSP 2018.
3. J. Kong, J. Kim, J. Bae. *HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis.* NeurIPS 2020.
4. D. Griffin, J. Lim. *Signal Estimation from Modified Short-Time Fourier Transform.* IEEE TASSP, 1984.
5. K. Ito, L. Johnson. *The LJ Speech Dataset.* 2017.
6. Y. Wang et al. *Tacotron: Towards End-to-End Speech Synthesis.* Interspeech 2017.
