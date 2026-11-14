"""Stage the Phase 3.8 private T4 engineering validation."""

import argparse
import ast
import json
from pathlib import Path

from prepare_kaggle_phase23 import ROOT, stage
from prepare_kaggle_phase36 import GPU_SETUP, RUNTIMES, SETUP, EVALUATION

TRAIN = """
for split, size in (("train", 671088640), ("validation", 67108864)):
    run("curl", "-fsSL", "--retry", "3", "-r", f"0-{size-1}", "-o", f"raw/{split}.prefix",
        f"https://open-images-dataset.s3.amazonaws.com/ocr/{split}.tgz")
run(cpu_python, "tools/prepare_phase38_data.py", "--raw", "raw", "--dictionary", dictionary,
    "--output", "phase38-data", "--retrieved-on", "2026-11-14")
(output / "training-audit.json").write_bytes((work / "phase38-data/audit.json").read_bytes())
for split in ("train", "validation"):
    (output / f"training-{split}.jsonl").write_bytes((work / f"phase38-data/{split}.jsonl").read_bytes())
run("uv", "venv", "--managed-python", "--python", "3.12.10", ".train-gpu")
interpreter = work / ".train-gpu/bin/python"
run("uv", "pip", "install", "--python", interpreter, "--index", "https://download.pytorch.org/whl/cu126",
    "-r", "requirements-torch.txt")
common = ["--dataset", "phase38-data", "--dictionary", dictionary, "--initial", "converted.safetensors",
          "--output", output / "training"]
run(interpreter, "tools/train_phase38.py", *common, "--stop-at-boundary")
run(interpreter, "tools/train_phase38.py", *common, "--resume")
"""

BACKWARD = """
device = DEVICES[0]
prior = next(p.parent for p in Path('/kaggle/input').rglob('capture.json') if p.parent.name == 'paddle')
common = ['--source', 'PaddleOCR', '--prior', prior, '--original', weights,
          '--checkpoint', 'converted.safetensors']
paddle_python, torch_python = runtime(device)
reference = output / 'paddle'
run(paddle_python, 'tools/phase38_backward.py', 'paddle', device, *common, '--output', reference)
run(torch_python, 'tools/phase38_backward.py', 'torch', device, *common,
    '--output', output / 'torch', '--reference', reference)
if device == 'gpu':
    archive = next(Path('/kaggle/input').rglob('real_parity.npz'))
    common = ['--source', 'PaddleOCR', '--archive', archive, '--original', weights,
              '--checkpoint', 'converted.safetensors']
    ref = output / 'training-reference-paddle'
    run(paddle_python, 'tools/phase38_training_reference.py', 'paddle', device, *common, '--output', ref)
    run(torch_python, 'tools/phase38_training_reference.py', 'torch', device, *common,
        '--output', output / 'training-reference-torch', '--reference', ref)
"""

REFERENCE = """
device = DEVICES[0]
archive = next(Path('/kaggle/input').rglob('real_parity.npz'))
common = ['--source', 'PaddleOCR', '--archive', archive, '--original', weights,
          '--checkpoint', 'converted.safetensors']
paddle_python, torch_python = runtime(device)
ref = output / 'paddle'
run(paddle_python, 'tools/phase38_training_reference.py', 'paddle', device, *common, '--output', ref)
run(torch_python, 'tools/phase38_training_reference.py', 'torch', device, *common,
    '--output', output / 'torch', '--reference', ref)
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--backward", choices=("cpu", "gpu"))
    parser.add_argument("--reference", choices=("cpu", "gpu"))
    args = parser.parse_args()
    stage(args.owner, args.directory, args.audit, True)
    prior = args.directory / "phase23.py"
    header = prior.read_text(encoding="utf8").split("\nimport json\n", 1)[0]
    files = ast.literal_eval(ast.parse(header).body[0].value)
    for file in (*ROOT.glob("src/ppocrv6_torch/**/*.py"),
                 *(ROOT / "tools" / (name + ".py") for name in
                   ("phase36_experiment", "phase36_evaluate", "trace_parity",
                    "prepare_phase38_data", "train_phase38", "phase38_backward", "trace_backward",
                    "phase38_training_reference"))):
        files[file.relative_to(ROOT).as_posix()] = file.read_text(encoding="utf8")
    runtime = RUNTIMES.replace('"safetensors==0.8.0")', '"safetensors==0.8.0", "accelerate==1.15.0")')
    device = args.backward or args.reference or "gpu"
    setup = SETUP if device == "gpu" else SETUP.replace('run("nvidia-smi", "-L")\n', "")
    code = (header + f"\nFILES = {files!r}\nDEVICES = {(device,)!r}\nGPU_SETUP = {GPU_SETUP!r}\n"
            + setup + '\ncpu_python = python\nos.environ.update(FLAGS_cudnn_deterministic="1", '
            'FLAGS_use_tf32="0", NVIDIA_TF32_OVERRIDE="0", CUBLAS_WORKSPACE_CONFIG=":4096:8")\n'
            + runtime + (BACKWARD if args.backward else REFERENCE if args.reference
                         else EVALUATION if args.inference else TRAIN))
    (args.directory / "phase38.py").write_text(code, encoding="utf8")
    prior.unlink()
    metadata_path = args.directory / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text())
    suffix = (f"backward-{device}" if args.backward else f"reference-{device}" if args.reference
              else "inference" if args.inference else "training")
    metadata.update(id=f"{args.owner}/indic-edge-ocr-phase38-{suffix}", title=f"Indic Edge OCR Phase38 {suffix}",
                    code_file="phase38.py", enable_gpu=device == "gpu",
                    kernel_sources=([f"{args.owner}/indic-edge-ocr-phase37-{device}/3"] +
                                    ([f"{args.owner}/indic-edge-ocr-phase23/4"] if device == "gpu" else []) if args.backward
                                    else [f"{args.owner}/indic-edge-ocr-phase23/4"] if args.reference else []))
    metadata_path.write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
