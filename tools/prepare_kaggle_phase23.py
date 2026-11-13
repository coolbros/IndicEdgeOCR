"""Stage the isolated Paddle/PyTorch comparison and optional GPU smoke job."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = """
import json
import os
from pathlib import Path
import subprocess

work = Path("/kaggle/temp/indic-phase23")
work.mkdir(parents=True, exist_ok=True)
output = Path("/kaggle/working")
for name, contents in FILES.items():
    path = work / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
os.environ.update(PYTHONPATH=str(work / "src"), PYTHONDONTWRITEBYTECODE="1",
                  NO_ALBUMENTATIONS_UPDATE="1", FLAGS_use_mkldnn="0",
                  OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                  CUDA_VISIBLE_DEVICES="0")

def run(*args, check=True):
    print("Running", str(args[0]), str(args[1]), flush=True)
    return subprocess.run(list(map(str, args)), cwd=work, check=check)

run("nvidia-smi", "-L")
run("uv", "venv", "--managed-python", "--python", "3.12.10", ".venv")
python = work / ".venv/bin/python"
run("uv", "pip", "install", "--python", python, "-r", "requirements-paddle.txt",
    "paddlepaddle==3.2.0", "safetensors==0.8.0")
run("uv", "pip", "install", "--python", python, "--index",
    "https://download.pytorch.org/whl/cpu", "torch==2.9.1+cpu")
run("git", "init", "PaddleOCR")
run("git", "-C", "PaddleOCR", "fetch", "--depth", "1",
    "https://github.com/PaddlePaddle/PaddleOCR.git", PADDLE_PIN)
run("git", "-C", "PaddleOCR", "checkout", "--detach", "FETCH_HEAD")
dictionary = "PaddleOCR/ppocr/utils/dict/ppocrv6_dict.txt"
weights = "PP-OCRv6_medium_rec_pretrained.pdparams"
run("curl", "-fsSL", "--retry", "3", "-o", weights,
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/" + weights)
(work / "raw").mkdir()
for split in ("train", "validation", "test"):
    run("curl", "-fsSL", "--retry", "3", "-o", f"raw/{split}.jsonl.gz",
        f"https://raw.githubusercontent.com/google-research-datasets/hiertext/{HIER_PIN}/gt/{split}.jsonl.gz")
    run("curl", "-fsSL", "--retry", "3", "-r", "0-8388607", "-o", f"raw/{split}.tgz.prefix",
        f"https://open-images-dataset.s3.amazonaws.com/ocr/{split}.tgz")
run(python, "tools/prepare_hiertext_subset.py", "--raw", "raw", "--output", "subset",
    "--dictionary", dictionary, "--retrieved-on", RETRIEVAL)
audit = json.loads((work / "subset/audit.json").read_text())
if audit["manifest_sha256"] != EXPECTED:
    raise ValueError("Unexpected split manifest checksums")
(output / "leakage-audit.json").write_text(json.dumps(audit, indent=2))
run(python, "tools/convert_paddle_checkpoint.py", weights, work / "converted.safetensors")
run(python, "tools/capture_paddle_parity.py", "--source", "PaddleOCR", "--checkpoint", weights,
    "--dataset", "subset", "--output", output / "paddle-real-cpu")
comparison = run(python, "tools/compare_torch_reference.py", "--original", weights,
    "--checkpoint", "converted.safetensors", "--archive", output / "paddle-real-cpu/real_operator_parity.npz",
    "--dataset", "subset", "--dictionary", dictionary,
    "--output", output / "pytorch-real-cpu.json", check=False)
if comparison.returncode not in (0, 1) or not (output / "pytorch-real-cpu.json").exists():
    raise RuntimeError("Parity comparison did not produce a report")
for path in (work / "converted.conversion.json",):
    (output / path.name).write_bytes(path.read_bytes())
if SMOKE:
    run("uv", "venv", "--managed-python", "--python", "3.12.10", ".venv-gpu")
    gpu_python = work / ".venv-gpu/bin/python"
    run("uv", "pip", "install", "--python", gpu_python, "--index",
        "https://download.pytorch.org/whl/cu126", "-r", "requirements-torch.txt")
    run(gpu_python, "tools/train_smoke.py", "--dataset", "subset",
        "--dictionary", dictionary, "--initial", "converted.safetensors",
        "--output", output / "smoke", "--precision", "fp16", "--stop-after-first-epoch")
    run(gpu_python, "tools/train_smoke.py", "--dataset", "subset",
        "--dictionary", dictionary, "--initial", "converted.safetensors",
        "--output", output / "smoke", "--precision", "fp16", "--resume")
"""


def stage(owner, directory, audit_path, smoke):
    paths = [
        *ROOT.glob("src/ppocrv6_torch/*.py"),
        ROOT / "requirements-paddle.txt",
        ROOT / "requirements-torch.txt",
        *(ROOT / "tools" / name for name in (
            "paddle_reference.py", "capture_paddle_parity.py", "compare_torch_reference.py",
            "convert_paddle_checkpoint.py", "prepare_hiertext_subset.py", "verify_dataset_splits.py",
        )),
    ]
    if smoke:
        paths.append(ROOT / "tools/train_smoke.py")
    files = {path.relative_to(ROOT).as_posix(): path.read_text(encoding="utf-8") for path in paths}
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit["passed"]:
        raise ValueError("Expected a passing split audit")
    directory.mkdir(parents=True, exist_ok=True)
    code = "phase23.py"
    headers = {"FILES": files, "SMOKE": smoke, "EXPECTED": audit["manifest_sha256"],
               "RETRIEVAL": "2026-11-13", "HIER_PIN": "70b6620b2b112597d8219e11eee9773a1403827c",
               "PADDLE_PIN": (ROOT / "references/paddleocr.lock").read_text().strip()}
    (directory / code).write_text("".join(f"{k} = {v!r}\n" for k, v in headers.items()) + RUN,
                                  encoding="utf-8")
    metadata = {"id": f"{owner}/indic-edge-ocr-phase23", "title": "Indic Edge OCR Phase23",
                "code_file": code, "language": "python", "kernel_type": "script",
                "is_private": True, "enable_gpu": True, "enable_internet": True,
                "dataset_sources": [], "competition_sources": [], "kernel_sources": []}
    (directory / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2),
                                                    encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    stage(args.owner, args.directory, args.audit, args.smoke)
