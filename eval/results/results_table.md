| System | WER | CER | Mel L1 | Mel RMSE |
|---|---|---|---|---|
| Real recording | 5.1% | 1.7% | - | - |
| Baseline (ours, no attention) + Griffin-Lim | 99.6% | 95.2% | 1.218 | 1.528 |
| Colatron (ours, from scratch) + Griffin-Lim | 28.6% | 12.1% | 0.873 | 1.151 |
| Colatron (ours, from scratch) + HiFi-GAN | 25.2% | 10.3% | 0.887 | 1.169 |
| Tacotron 2 (pretrained) + Griffin-Lim | 17.6% | 8.7% | 0.878 | 1.175 |
| Tacotron 2 (pretrained) + HiFi-GAN | 16.2% | 7.8% | 0.901 | 1.206 |
