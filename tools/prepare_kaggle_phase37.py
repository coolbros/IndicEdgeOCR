"""Stage a private paired Phase 3.7 Kaggle run."""

import argparse
import ast
import json
from pathlib import Path

from prepare_kaggle_phase23 import ROOT, stage
from prepare_kaggle_phase36 import GPU_SETUP, RUNTIMES, SETUP


RUN = """
os.environ.update(FLAGS_cudnn_deterministic="1", FLAGS_use_tf32="0",
                  NVIDIA_TF32_OVERRIDE="0", CUBLAS_WORKSPACE_CONFIG=":4096:8")
prior = next(p for p in Path("/kaggle/input").rglob("real_parity.npz")
             if p.with_name("manifest.json").is_file())
common = ["--archive", prior, "--original", weights, "--checkpoint",
          "converted.safetensors", "--dataset", "subset", "--source", "PaddleOCR"]
paddle_python, torch_python = runtime(DEVICE)
paddle_output = output / "paddle"
torch_output = output / "torch"
run(paddle_python, "tools/trace_backward.py", "paddle", DEVICE, *common, "--output", paddle_output)
run(torch_python, "tools/trace_backward.py", "torch", DEVICE, *common,
    "--reference", paddle_output, "--output", torch_output)
if DEVICE == "cpu":
    for variant, flags in (("one-thread", ["--threads", "1"]), ("no-mkldnn", ["--no-mkldnn"])):
        run(torch_python, "tools/trace_backward.py", "torch", DEVICE, *common,
            "--reference", paddle_output, "--output", output / variant, "--metrics-only", *flags)
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--reproduce-phase36", action="store_true")
    args = parser.parse_args()
    stage(args.owner, args.directory, args.audit, False)
    prior = args.directory / "phase23.py"
    header = prior.read_text(encoding="utf8").split("\nimport json\n", 1)[0]
    files = ast.literal_eval(next(node.value for node in ast.parse(header).body
                                  if node.targets[0].id == "FILES"))
    for path in (*ROOT.glob("src/ppocrv6_torch/ops/*.py"),
                 *(ROOT / "tools" / name for name in ("trace_backward.py", "trace_parity.py",
                                                       "phase36_experiment.py"))):
        files[path.relative_to(ROOT).as_posix()] = path.read_text(encoding="utf8")
    setup = SETUP if args.device == "gpu" else SETUP.replace('run("nvidia-smi", "-L")\n', "")
    script = (header + f"\nFILES = {files!r}\nDEVICE = {args.device!r}\nGPU_SETUP = {GPU_SETUP!r}\n"
              + setup + "\ncpu_python = python\n" + RUNTIMES + RUN)
    if args.reproduce_phase36:
        script += '''
p = output / "phase36-paddle"
run(paddle_python, "tools/phase36_experiment.py", "paddle", DEVICE, *common, "--output", p)
run(torch_python, "tools/phase36_experiment.py", "torch", DEVICE, *common,
    "--reference", p / "reference.npz", "--output", output / "phase36-torch")
'''
    name = f"indic-edge-ocr-phase37-{args.device}"
    (args.directory / "phase37.py").write_text(script, encoding="utf8")
    prior.unlink()
    metadata_path = args.directory / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(id=f"{args.owner}/{name}", title=f"Indic Edge OCR Phase37 {args.device}",
                    code_file="phase37.py", enable_gpu=args.device == "gpu",
                    kernel_sources=[args.reference])
    metadata_path.write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
