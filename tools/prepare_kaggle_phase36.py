"""Stage matching Paddle/PyTorch experiments without modifying prior runs."""

import argparse
import ast
import json
from pathlib import Path

from prepare_kaggle_phase23 import ROOT, RUN, stage
from prepare_kaggle_reference import RUNNER


SETUP = RUN.split('run(python, "tools/capture_paddle_parity.py"')[0]
GPU_SETUP = RUNNER[RUNNER.index('wheel = '):RUNNER.index('run("git", "init"')]
RUNTIMES = """
import shutil
def runtime(device):
    if device == "gpu":
        from concurrent.futures import ThreadPoolExecutor
        run("uv", "venv", "--managed-python", "--python", "3.12.10", ".paddle-gpu")
        run("uv", "venv", "--managed-python", "--python", "3.12.10", ".torch-gpu")
        paddle_python = work / ".paddle-gpu/bin/python"
        torch_python = work / ".torch-gpu/bin/python"
        run("uv", "pip", "install", "--python", paddle_python, "-r", "requirements-paddle.txt")
        exec(GPU_SETUP, dict(globals(), uv="uv", python=paddle_python,
                             ThreadPoolExecutor=ThreadPoolExecutor))
        run("uv", "pip", "install", "--python", torch_python, "--index",
            "https://download.pytorch.org/whl/cu126", "torch==2.9.1+cu126",
            "numpy==1.26.4", "opencv-python-headless==4.11.0.86", "safetensors==0.8.0")
        # Paddle environment imports the CPU torch comparison helper, never CUDA torch.
        paddle_site = subprocess.check_output(
            [paddle_python, "-c", "import site; print(site.getsitepackages()[0])"],
            text=True).strip()
        cpu_torch_site = subprocess.check_output(
            [cpu_python, "-c", "import site; print(site.getsitepackages()[0])"],
            text=True).strip()
        (Path(paddle_site) / "cpu-torch.pth").write_text(cpu_torch_site + "\\n")
    else:
        paddle_python = torch_python = cpu_python
    return paddle_python, torch_python
"""

EXPERIMENTS = """
prior = next(p for p in Path("/kaggle/input").rglob("real_parity.npz")
             if p.with_name("manifest.json").is_file())
common = ["--archive", prior, "--original", weights, "--checkpoint",
          "converted.safetensors", "--dataset", "subset", "--source", "PaddleOCR"]
for device in DEVICES:
    paddle_python, torch_python = runtime(device)
    p = output / f"paddle-{device}"
    run(paddle_python, "tools/phase36_experiment.py", "paddle", device, *common, "--output", p)
    run(torch_python, "tools/phase36_experiment.py", "torch", device, *common,
        "--reference", p / "reference.npz", "--output", output / f"torch-{device}")
"""

EVALUATION = """
run("curl", "-fsSL", "--retry", "3", "-r", "0-67108863", "-o", "raw/validation.prefix",
    "https://open-images-dataset.s3.amazonaws.com/ocr/validation.tgz")
data = output / "validation"
run(cpu_python, "tools/phase36_evaluate.py", "prepare", "--annotations", "raw/validation.jsonl.gz",
    "--prefix", "raw/validation.prefix", "--dataset", "subset", "--output", data,
    "--retrieved-on", RETRIEVAL)
device = DEVICES[0]
for framework, interpreter in zip(("paddle", "torch"), runtime(device)):
    run(interpreter, "tools/phase36_evaluate.py", "evaluate", "--framework", framework,
        "--device", device, "--dataset", data, "--original", weights,
        "--checkpoint", "converted.safetensors", "--source", "PaddleOCR",
        "--output", output / f"predictions-{framework}.json")
run(cpu_python, "tools/phase36_evaluate.py", "compare", "--paddle", output / "predictions-paddle.json",
    "--torch", output / "predictions-torch.json", "--output", output / "recognition.json")
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--device", choices=("cpu", "gpu", "both"), default="both")
    parser.add_argument("--evaluation", action="store_true")
    parser.add_argument("--decoder-only", action="store_true")
    args = parser.parse_args()
    stage(args.owner, args.directory, args.audit, False)
    path = args.directory / "phase23.py"
    code = path.read_text(encoding="utf-8")
    header = code.split("\nimport json\n", 1)[0]
    files = ast.literal_eval(next(node.value for node in ast.parse(header).body
                                  if node.targets[0].id == "FILES"))
    for file in (*ROOT.glob("src/ppocrv6_torch/ops/*.py"),
                 ROOT / "tools/trace_parity.py", ROOT / "tools/phase36_experiment.py",
                 ROOT / "tools/phase36_evaluate.py"):
        files[file.relative_to(ROOT).as_posix()] = file.read_text(encoding="utf-8")
    devices = ("cpu", "gpu") if args.device == "both" else (args.device,)
    if args.evaluation:
        devices = ("gpu",) if args.device == "gpu" else ("cpu",)
    script = (header + f"\nFILES = {files!r}\nDEVICES = {devices!r}\nGPU_SETUP = {GPU_SETUP!r}\n"
              + SETUP + '\ncpu_python = python\nos.environ.update(FLAGS_cudnn_deterministic="1", '
              'FLAGS_use_tf32="0", NVIDIA_TF32_OVERRIDE="0", CUBLAS_WORKSPACE_CONFIG=":4096:8")\n'
              + RUNTIMES + (EVALUATION if args.evaluation else EXPERIMENTS))
    if "gpu" not in devices:
        script = script.replace('run("nvidia-smi", "-L")\n', "")
    if args.decoder_only:
        script = script.replace('common = ["--archive", prior',
                                'common = ["--decoder-only", "--archive", prior')
    (args.directory / "phase36.py").write_text(script, encoding="utf-8")
    path.unlink()
    metadata_path = args.directory / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text())
    suffix = "eval" if args.evaluation else args.device
    if args.evaluation and devices == ("gpu",):
        suffix += "-gpu"
    if args.decoder_only:
        suffix += "-decoder"
    metadata.update(id=f"{args.owner}/indic-edge-ocr-phase36-{suffix}",
                    title=f"Indic Edge OCR Phase36 {suffix}", code_file="phase36.py",
                    enable_gpu="gpu" in devices,
                    kernel_sources=[] if args.evaluation else [args.reference])
    metadata_path.write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
