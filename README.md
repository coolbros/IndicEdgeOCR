# Indic Edge OCR

Faithful reproduction of the official PP-OCRv6 medium recognizer, starting with an architecture audit and executable Paddle baseline.

- [Architecture and operator audit](reports/architecture_audit.md)
- [Verified Paddle baseline](reports/paddle_reference.md)
- [Phase checklist](TODO.md)

## Kaggle Paddle reference

Stage the private job and use the official Kaggle CLI with `KAGGLE_API_TOKEN` configured in the invoking environment.

```bash
python tools/prepare_kaggle_reference.py --owner YOUR_KAGGLE_USERNAME --directory artifacts/kaggle
kaggle kernels push -p artifacts/kaggle --accelerator NvidiaTeslaT4 --timeout 2700
kaggle kernels status YOUR_KAGGLE_USERNAME/indic-edge-ocr-paddle-reference
kaggle kernels output YOUR_KAGGLE_USERNAME/indic-edge-ocr-paddle-reference -p artifacts/paddle
```

Kaggle requests two T4s and the reference captures from `gpu:0`. The job creates an isolated Python 3.12.10 environment with PaddlePaddle GPU 3.2.0 (CUDA 12.6). It fetches the pinned, clean PaddleOCR checkout, the official checkpoint and the official sample. The checkpoint SHA-256 is validated before loading.

The output is `reference.npz` plus `manifest.json`. NPZ keys start with `eval/official`, `eval/synthetic_0`, `eval/synthetic_1`, or `train`. The archive contains model inputs, backbone and neck activations, CTC logits and probabilities, teacher-forced NRTR logits, encoded training labels, and loss scalars. The manifest records decoded text and confidence, per-layer shapes, parameter counts, and source and artifact provenance. The training fixture uses the official fixed-size labeled preprocessing without stochastic image augmentation. The forward pass uses training-mode BatchNorm and dropout with a fixed seed.

## Run against a local Paddle environment

With Python 3.12, clone and pin the upstream checkout, and install PaddlePaddle 3.2.0 in an isolated environment.

```bash
git clone --filter=blob:none --no-checkout https://github.com/PaddlePaddle/PaddleOCR.git references/PaddleOCR
git -C references/PaddleOCR checkout --detach dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf
python -m venv .venv-paddle
source .venv-paddle/bin/activate
python -m pip install -r requirements-paddle.txt paddlepaddle==3.2.0
mkdir -p weights/original references/samples
curl -fL https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_medium_rec_pretrained.pdparams -o weights/original/PP-OCRv6_medium_rec_pretrained.pdparams
curl -fL https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_rec_001.png -o references/samples/general_ocr_rec_001.png
python tools/paddle_reference.py \
  --checkpoint weights/original/PP-OCRv6_medium_rec_pretrained.pdparams \
  --sample references/samples/general_ocr_rec_001.png
python -m unittest discover -s tests
```

For local CUDA execution, use Paddle's GPU wheel and pass `--device gpu:0`. [PaddlePaddle installation](https://www.paddleocr.ai/main/en/version3.x/paddlepaddle_installation.html).

## Phase 2/3

- [Checkpoint conversion and parity results](reports/pytorch_parity.md)
- [Dataset provenance and preparation](reports/dataset_provenance.md)
- [Split audit and manifest hashes](reports/data_leakage_audit.md)

Install `requirements-torch.txt` in an isolated Python 3.12 environment and set
`PYTHONPATH=src`. The parity report includes exact local conversion and comparison
commands. Dataset preparation downloads the official partitions, creates a new
subset directory, and records split and crop checksums.

```bash
python tools/prepare_kaggle_phase23.py --owner YOUR_KAGGLE_USERNAME \
  --audit artifacts/hiertext/subset/audit.json --directory artifacts/kaggle-phase23 --smoke
kaggle kernels push -p artifacts/kaggle-phase23 --accelerator NvidiaTeslaT4 --timeout 2100
kaggle kernels status YOUR_KAGGLE_USERNAME/indic-edge-ocr-phase23
kaggle kernels output YOUR_KAGGLE_USERNAME/indic-edge-ocr-phase23 -p artifacts/phase23-kaggle
```

The private job rebuilds and verifies the input manifests and runs an isolated
CPU FP32 comparison. It records differences in JSON. With `--smoke`, it runs
Accelerate on one CUDA device, saves `latest` and validation-selected `best`
with SHA-256 metadata, and starts a new process to resume `latest`. Deterministic
preprocessed inputs are cached by image digest. The smoke job uses a fixed batch
without image augmentation.

Run held-out evaluation as a separate command after freezing model selection.

```bash
PYTHONPATH=src python tools/evaluate_heldout.py \
  --dataset artifacts/hiertext/subset \
  --dictionary references/PaddleOCR/ppocr/utils/dict/ppocrv6_dict.txt \
  --checkpoint artifacts/phase23-kaggle/smoke/best.pt \
  --output artifacts/heldout.json
```
