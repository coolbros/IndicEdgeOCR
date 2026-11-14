# Phase 3.8 bounded training validation

The private Kaggle job uses an official HierText train/validation partition subset for engineering validation. The source is Open Images via the HierText publisher. Annotations are CC BY-SA 4.0 and underlying photographs retain source attribution and applicable terms. HierText is independent of the PP-OCRv6 pretraining recipe. No test examples enter training or validation. [HierText release](https://github.com/google-research-datasets/hiertext).

Preparation reads the first 640 MiB of the official training archive and the first 64 MiB of validation. It retains crops with legible horizontal labels of at most 23 code points that the frozen dictionary encodes in full and at least 10 pixels on each side. The prepared audit retained 5,198 training and 256 validation crops. Source-image IDs, source and image SHA-256, and dHash checks use the existing auditor. The validation list is fixed first and the flagged training crops are excluded. Audit metrics include counts by source, character and length.

| Manifest | SHA-256 |
|---|---|
| Train | `edf26d9969719c2dd4026cd1245002c59699d56e16031363aef830e3b335c5d0` |
| Validation | `bd15ac59ec8d982cde7184e36d90f7f8e227ceec6f760c52e12fdb5be3d5e4c7` |

## Commands

```bash
python tools/prepare_kaggle_phase38.py --owner agastyavar \
  --directory artifacts/phase38/training-job --audit artifacts/hiertext/subset/audit.json
kaggle kernels push -p artifacts/phase38/training-job --accelerator NvidiaTeslaT4 --timeout 7200
kaggle kernels output agastyavar/indic-edge-ocr-phase38-training/2 -p artifacts/phase38/training-v2 \
  --file-pattern '(audit|results|latest|best)[.]json$|[.]jsonl$' --page-size 50
python -m unittest discover -s tests
```

Stage the independent frozen inference rerun with `--inference`, the primitive replays with `--backward cpu` or `--backward gpu`, and the optimizer and loss reference with `--reference cpu` or `--reference gpu`. Use `--accelerator NvidiaTeslaT4` for GPU jobs. The runner uses just one GPU of the allocated T4 pair.

The run starts from the frozen converted checkpoint and uses Accelerate in FP16 while the loss computations upcast logits to FP32. The run uses 51 optimizer steps per epoch for two epochs, with five warmup steps and an engineering LR of 5e-5. Excluding the initial zero-LR warmup step leaves 101 parameter-update opportunities. The runner saves validation-selected best and latest, exits at the first boundary, then resumes in a fresh process and verifies the global cursor and Adam state. Per-update metrics record both recognition losses, LR, gradient norms, AMP overflow retries and multi-scale canvas.

## T4 execution and checkpoint evidence

[Training version 2](https://www.kaggle.com/code/agastyavar/indic-edge-ocr-phase38-training/versions/2) completed 102 successful optimizer steps, 101 with nonzero LR, and restored the hash-verified first checkpoint at step 51 in a fresh process. All 318 Adam parameter-state entries resumed at the saved step. A CPU unit test separately reproduces the next update and first moment exactly after a restart. The GPU run validates actual separate-process restoration, without an uninterrupted full-model comparison run.

| Measure | Observed |
|---|---:|
| Validation exact matches before training | 219 / 256 |
| Validation exact matches at step 51 | 177 / 256 |
| Validation exact matches at step 102 | 189 / 256 |
| Scaffold and resume training-loop wall time | 46.09 s + 51.03 s |
| Largest CUDA allocation | 1,393,460,736 bytes |
| Step counts at heights 32, 48, 64 | 37, 31, 34 |
| Skipped AMP overflow attempts, retried | 1 |
| First and final successful batch total loss | 5.117 and 1.696 |
| First and last 10-batch median total loss | 10.723 and 11.268 |
| First and final pre-step learning rate | 0 and 4.257e-7 |

Losses and tracked gradient norms were finite for every successful update. Final L2 parameter changes for stem, CTC projection and NRTR projection were 0.01216, 1.4100 and 1.5384. Across successful batches their gradient norm ranges were [6.12, 283.45], [1.27, 86.75] and [0.864, 5.98]. Batch losses vary with crop content and augmentation. Validation exact match declined from 85.55% to 73.83%. The run demonstrates that the pipeline updates, resumes and evaluates, without an accuracy improvement claim.

The audited crops come from 61 training source images and six distinct validation source images. The final audit found no exact or near-duplicate crop across the two splits after excluding two flagged training examples. Before reaching 5,200 training candidates, the selector rejected 2,471 for orientation or legibility, 1,270 for size and 186 for unsupported or overlong labels. Corresponding validation rejections were 43, 169 and four. The private audit JSON contains the complete character and word-length distributions.

| Artifact | SHA-256 |
|---|---|
| Training audit | `6fb3180fdddc6d0ffa8298497c807fcf9359794108ffac3e57982e10d4d0fdac` |
| Per-step results, resumed | `fdd4c8d68cf8c7147db2adaedbf227272bcc027cf84a6606d56510d80b5f3c9a` |
| Step 51 latest, verified at resume | `068f7ddf6f6fa59624aa780cb595a2b3981105c08ab652e5681c9e048cae131d` |
| Step 102 latest | `4c39222c05f5451005f5e978e27382387d3092e529c9ed8ec13dfd44598e387d` |
| Step 102 best | `6f9243470c0f276d2d21e26c66719b1d91a437cb641598483cf1497fe620bb65` |

Runtime was one Tesla T4, Torch 2.9.1+cu126, CUDA 12.6, cuDNN 9.10.2 and Accelerate 1.15.0. The official checkpoint SHA was rechecked after both processes. The split manifests, checkpoint sidecars and per-step metrics are also available under the ignored `artifacts/phase38/` directory.

The published 677,305,063-byte `latest.pt` was also downloaded, verified against its sidecar SHA-256 and loaded strictly into a fresh default model on CPU. Its scheduler and all 318 Adam state entries report step 102.
