# CRES-Net
CRES-Net：Cover‑Referenced Evidence Steganalysis Network
You can find the dataset at the following Google Drive link: [datalink](https://drive.google.com/drive/folders/152NGEbKxxzWfQhowh_6ufyHuXOznqisU)

1、Recommended environment:
- Python 3.10+
- PyTorch 2.x
- NumPy
- pandas
- scikit-learn
- NVIDIA GPU with CUDA support

2、Reported Results
Mean accuracy (%) across repeated runs:
| Setting | Rate | CRES-Net | DVSF |
|---|---:|---:|---:|
| 1-s | 10% | 82.80 | 79.03 |
| 1-s | 20% | 94.46 | 91.75 |
| 1-s | 30% | 97.38 | 96.88 |
| 1-s | 40% | 98.09 | 98.72 |
| 1-s | 50% | 98.11 | 99.27 |
| 0.1-s | 10% | 60.75 | 58.14 |
| 0.1-s | 20% | 68.62 | 66.02 |
| 0.1-s | 30% | 74.87 | 72.56 |
| 0.1-s | 40% | 79.42 | 78.09 |
| 0.1-s | 50% | 82.85 | 83.71 |

3、Evaluate a trained checkpoint with:
python CRES-Net.py \
  --test-only \
  --checkpoint path/to/model_best.pth.tar \
  --test-steg path/to/Steg \
  --test-cover path/to/Cover \
  --test-rate 10
