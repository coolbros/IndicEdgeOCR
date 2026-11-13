"""Stage the CPU-only parity trace using the private version-4 output."""

import argparse
import ast
import json
from pathlib import Path

from prepare_kaggle_phase23 import ROOT, RUN, stage


SETUP = RUN.split('run(python, "tools/capture_paddle_parity.py"')[0]
SETUP = SETUP.replace('run("nvidia-smi", "-L")\n', '')
TRACE = """
import hashlib
import platform

prior = next(p for p in Path("/kaggle/input").rglob("real_parity.npz")
             if p.with_name("manifest.json").is_file())
def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

for name, path in (("archive", prior), ("original", work / weights),
                   ("checkpoint", work / "converted.safetensors"),
                   ("train_manifest", work / "subset/train.jsonl")):
    if digest(path) != BASELINE_HASHES[name]:
        raise ValueError(f"Baseline checksum mismatch for {name}")
baseline = run(python, "tools/compare_torch_reference.py", "--original", weights,
    "--checkpoint", "converted.safetensors", "--archive", prior, "--dataset", "subset",
    "--dictionary", dictionary, "--output", output / "baseline.json", check=False)
if baseline.returncode not in (0, 1) or not (output / "baseline.json").exists():
    raise RuntimeError("Baseline comparison failed to run")
cpu = Path("/proc/cpuinfo").read_text().split("\\n\\n")[0]
(output / "host.json").write_text(json.dumps({"platform": platform.platform(),
    "cpu": cpu, "python": str(subprocess.check_output([python, "--version"], text=True).strip())}))
common = ["--archive", prior, "--original", weights,
          "--checkpoint", "converted.safetensors", "--dataset", "subset"]
boundaries = output / "paddle-boundaries"
run(python, "tools/trace_parity.py", "capture", *common,
    "--source", "PaddleOCR", "--output", boundaries)
for mkldnn, threads in (("on", 2), ("off", 2), ("on", 1)):
    dest = output / f"replay-{mkldnn}-{threads}"
    run(python, "tools/trace_parity.py", "replay", *common, "--boundaries", boundaries,
        "--mkldnn", mkldnn, "--threads", threads, "--output", dest,
        *(["--metrics-only"] if (mkldnn, threads) != ("on", 2) else []))
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--t4-host", action="store_true",
                        help="Request the same pool as v4, while running every calculation on CPU")
    args = parser.parse_args()
    stage(args.owner, args.directory, args.audit, False)
    job = args.directory / "phase23.py"
    header = job.read_text(encoding="utf-8").split("\nimport json\n", 1)[0]
    trace = (ROOT / "tools/trace_parity.py").read_text(encoding="utf-8")
    assignment = next(node for node in ast.parse(trace).body if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "BASELINE" for t in node.targets))
    (args.directory / "phase35.py").write_text(
        header + f"\nFILES['tools/trace_parity.py'] = {trace!r}\n"
        f"BASELINE_HASHES = {ast.literal_eval(assignment.value)!r}\n" + SETUP + TRACE,
        encoding="utf-8")
    job.unlink()
    path = args.directory / "kernel-metadata.json"
    metadata = json.loads(path.read_text())
    metadata.update(id=f"{args.owner}/indic-edge-ocr-phase35",
                    title="Indic Edge OCR Phase35", code_file="phase35.py",
                    enable_gpu=args.t4_host,
                    kernel_sources=[args.reference])
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
