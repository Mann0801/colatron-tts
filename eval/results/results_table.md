| System | WER | CER | Mel L1 | Mel RMSE |
|---|---|---|---|---|
| Real recording | 5.1% | 1.7% | - | - |
| Baseline (ours, no attention) + Griffin-Lim | 99.8% | 94.8% | 1.217 | 1.527 |
| Colatron (ours, from scratch) + Griffin-Lim | 9.3% | 3.0% | 0.816 | 1.081 |
| Colatron (ours, from scratch) + HiFi-GAN | 8.8% | 2.8% | 0.832 | 1.101 |
| Tacotron 2 (pretrained) + Griffin-Lim | 17.6% | 8.7% | 0.878 | 1.175 |
| Tacotron 2 (pretrained) + HiFi-GAN | 16.2% | 7.8% | 0.901 | 1.206 |
