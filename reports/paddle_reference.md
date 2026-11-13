# Verified Paddle reference

The reference ran in FP32 on `gpu:0` with PaddlePaddle GPU 3.2.0, CUDA 12.6, Python 3.12.10, NumPy 1.26.4, OpenCV 4.11.0, and seed 2026. Kaggle allocated two Tesla T4s. The job captured a single-process forward pass on the first T4. [Kaggle run version 3](https://www.kaggle.com/code/agastyavar/indic-edge-ocr-paddle-reference/versions/3).

The official pretrained loader loaded all 441 expected tensors. The runner verified that every loaded tensor equaled the corresponding checkpoint tensor and found zero missing, unexpected, or mismatched tensors. The runtime inventory matched the audit's backbone, LightSVTR, CTC and NRTR counts. Total trainable parameters were **55,527,798**.

| Sample | Input NCHW | CTC time steps | Decoded text | Confidence |
|---|---|---:|---|---:|
| Official | 1 × 3 × 48 × 385 | 48 | 绿洲仕格维花园公寓 | 0.99962914 |
| Synthetic 0 | 1 × 3 × 48 × 320 | 40 | AB123 | 0.99999464 |
| Synthetic 1 | 1 × 3 × 48 × 365 | 46 | XYZ | 0.99806976 |

The fixed-size labeled batch used both synthetic images. Its input shape was `[2,3,48,320]`, with CTC logits `[2,40,18710]` and teacher-forced NRTR logits `[2,6,18714]`.

| Loss | Value |
|---|---:|
| CTC | 0.3714269102 |
| NRTR | 1.6743669510 |
| Total | 2.0457937717 |

The Kaggle output and local `artifacts/paddle/` contain `reference.npz` and `manifest.json`. The archive SHA-256 is `d07c2d3eb5f31656a462e150bbc47e802dcd718bf3fc835d0b8464fdb0dc5bb5`. The JSON includes checkout and checkpoint provenance, decodes, module inventory, and per-sample runtime shapes. Large files are ignored by Git.

The artifact test checks the expected transcription, vocabulary dimensions, checkpoint coverage, nominal and wide preprocessing, intermediate fixture presence, finite tensor values, and the loss sum. The initial GPU jobs hit Kaggle's Python 3.13 environment bootstrap and Paddle's implicit setuptools dependency. The successful job used managed Python 3.12.10, explicit setuptools, and Paddle's official wheel origin with range downloads. The production inference archive was inspected statically. The executed baseline uses the official dynamic model initialized from the training checkpoint. Local verification checks the downloaded artifacts. Local Windows application control blocked loading the Paddle native extension.

Reproduction and download commands are in the [README](../README.md).
