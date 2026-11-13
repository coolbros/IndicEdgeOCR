"""Stage the private Kaggle GPU job from the checked-out project files."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = """
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import shutil

work = Path("/kaggle/temp/ppocrv6-reference")
work.mkdir(parents=True, exist_ok=True)
output = Path("/kaggle/working")
venv = work / ".venv"
python = venv / "bin" / "python"
base = "https://paddle-model-ecology.bj.bcebos.com/paddlex"
model = "PP-OCRv6_medium_rec_pretrained.pdparams"
sample = "general_ocr_rec_001.png"

def run(*args):
    subprocess.run(list(map(str, args)), cwd=work, check=True)

for name, content in FILES.items():
    (work / name).write_text(content, encoding="utf-8")

run("nvidia-smi", "-L")
uv = shutil.which("uv")
if uv is None:
    release = "https://github.com/astral-sh/uv/releases/download/0.9.30"
    archive = "uv-x86_64-unknown-linux-gnu.tar.gz"
    run("curl", "-fsSL", "--retry", "3", "-o", archive, f"{release}/{archive}")
    run("tar", "-xzf", archive)
    uv = work / "uv-x86_64-unknown-linux-gnu" / "uv"
run(uv, "venv", "--managed-python", "--python", "3.12.10", venv)
run(uv, "pip", "install", "--python", python, "-r", work / "requirements-paddle.txt")
wheel = "paddlepaddle_gpu-3.2.0-cp312-cp312-linux_x86_64.whl"
wheel_origin = "https://paddle-whl.bj.bcebos.com/stable/cu126/paddlepaddle-gpu"
wheel_size = 1_890_365_820
chunk_size = 64 * 1024 * 1024

def fetch_part(start):
    stop = min(start + chunk_size, wheel_size) - 1
    part = work / f"paddle-{start:012d}.part"
    run("curl", "-fsSL", "--retry", "3", "--max-time", "240",
        "--range", f"{start}-{stop}", "-o", part, f"{wheel_origin}/{wheel}")
    if part.stat().st_size != stop - start + 1:
        raise ValueError(f"Unexpected wheel range size for {start}-{stop}")
    print(f"Fetched Paddle wheel bytes {start}-{stop}", flush=True)
    return part

with ThreadPoolExecutor(max_workers=12) as pool:
    parts = list(pool.map(fetch_part, range(0, wheel_size, chunk_size)))
with (work / wheel).open("wb") as dest:
    for part in parts:
        with part.open("rb") as stream:
            shutil.copyfileobj(stream, dest)
        part.unlink()
run(uv, "pip", "install", "--python", python, work / wheel,
    "--index", "https://www.paddlepaddle.org.cn/packages/stable/cu126/")
run("git", "init", "PaddleOCR")
run("git", "-C", "PaddleOCR", "fetch", "--depth", "1",
    "https://github.com/PaddlePaddle/PaddleOCR.git", PIN)
run("git", "-C", "PaddleOCR", "checkout", "--detach", "FETCH_HEAD")
run("curl", "-fsSL", "--retry", "3", "-o", model,
    f"{base}/official_pretrained_model/{model}")
run("curl", "-fsSL", "--retry", "3", "-o", sample,
    f"{base}/imgs/demo_image/{sample}")
os.environ.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1",
                  FLAGS_cudnn_deterministic="1", FLAGS_use_tf32="0",
                  PADDLE_REFERENCE_OUTPUT=str(output), NO_ALBUMENTATIONS_UPDATE="1")
run(python, "paddle_reference.py", "--source", "PaddleOCR", "--checkpoint", model,
    "--sample", sample, "--output", output, "--device", "gpu:0")
run(python, "-m", "unittest", "test_paddle_reference")
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    code_file = "paddle_reference_job.py"
    files = {
        "paddle_reference.py": "tools/paddle_reference.py",
        "requirements-paddle.txt": "requirements-paddle.txt",
        "test_paddle_reference.py": "tests/test_paddle_reference.py",
    }
    embedded = {name: (ROOT / path).read_text(encoding="utf-8")
                for name, path in files.items()}
    pin = (ROOT / "references/paddleocr.lock").read_text().strip()
    source = f"PIN = {pin!r}\nFILES = {embedded!r}\n" + RUNNER
    (args.directory / code_file).write_text(source, encoding="utf-8")
    metadata = {
        "id": f"{args.owner}/indic-edge-ocr-paddle-reference",
        "title": "Indic Edge OCR Paddle Reference",
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (args.directory / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
