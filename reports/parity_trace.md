# Phase 3.5 FP32 parity trace

This trace starts at project commit `27928fed9fa5b1c14ce19a8accf446c7de5ead10` and PaddleOCR commit `dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf`. The v4 fixture uses four real HierText training images. The replay has one real inference sample and the full four-image training-forward batch. Dropout is disabled in the latter and BatchNorm stays in training mode. Neither the model nor checkpoint was changed.

## Baseline accounting

| Run | Tensor comparisons | Decode comparisons | Passed |
|---|---:|---:|---:|
| Phase 2/3 v2 | 976 | 4 | 555 / 980 |
| Phase 2/3 v4 | 986 | 4 | 565 / 990 |
| Phase 3.5 Windows CPU with v4 inputs | 986 | 4 | 552 / 990 |
| Phase 3.5 Kaggle CPU-only worker with v4 inputs | 986 | 4 | 490 / 990 |
| Phase 3.5 CPU on original T4 worker pool with v4 inputs | 986 | 4 | 565 / 990 |

The v4 report adds exactly ten passing operator probes. Every v2 comparison retains its original pass status in v4. A tensor row passes when shape and dtype match and every element satisfies `abs(actual - expected) <= atol + rtol * abs(expected)`. Decoding rows use exact text and the original confidence tolerance. Counts are per tensor or decode and do not weight large tensors more heavily. The replay also records matching signed infinities in causal masks and excludes those positions from finite-error summaries. [Original v4 run](https://www.kaggle.com/code/agastyavar/indic-edge-ocr-phase23/versions/4).

## NRTR memory injection on Windows

The native memory projection with natural input missed the tolerance, with maximum absolute error 1.769e-4. With the original Paddle pooled backbone tensor injected, the projection passed all 81,920 elements, with maximum absolute error 1.144e-5. Contiguous FP32, strided FP32, and the FP64 accumulation diagnostic had the same maximum error.

| Decoder output | Max error with exact Paddle memory | Outliers |
|---|---:|---:|
| Block 0 cross-attention KV projection | 2.146e-6 | 0 |
| Block 0 cross-attention normalization | 5.811e-7 | 0 |
| Block 0 output | 4.530e-6 | 0 |
| Block 1 output | 1.335e-5 | 0 |
| Block 2 output | 1.830e-5 | 0 |
| Block 3 output | 4.315e-5 | 5 |
| Final NRTR logits | 3.958e-5 | 11 |

Exact memory injection reduces the Windows NRTR logit maximum from 4.624e-3 to 3.958e-5. This implicates propagated backbone error as the dominant contribution. A smaller decoder-local accumulation remains at the declared tolerance.

On the Windows decoder-only replay, disabling MKLDNN at two threads left the result unchanged. Using one thread reduced the final maximum from 3.958e-5 to 3.815e-5 and outliers from 11 to 8. These runs kept identical memory and tokens and retained the FP32 graph.

| Immutable input | SHA-256 |
|---|---|
| v4 Paddle NPZ | `e861cbf75fe498d05e4edc0fa568a09cabe88f6224b2e38427ee557df65af8fc` |
| v4 comparison JSON | `f635a681b652209c854b22b23a8e00d246cb31a511d176c331d70b562d28f28b` |
| HierText training JSONL | `4fc83fe5c90366051e75d2f10ef61624a68dbec3785e263a1bcd14042cdfed73` |
| Official Paddle checkpoint | `c4a110423c23739f157ff870e27f41649d8c89e4c8df5230c2cd2e9fd90a7e92` |
| Converted PyTorch checkpoint | `0afc351e31cea4f6d670e68652bd22a11ab44273584e156159dbdb83fafb02ef` |

## Paired replay on Kaggle

The Intel Xeon worker in the original T4 pool reproduced the v4 comparison JSON byte for byte. Fresh Paddle inference probabilities and teacher-forced logits also reproduced their v4 tensors exactly. The boundary trace was run on CPU. The default run contains 361 inference and 517 training rows, with 870 of 878 exact-input replays passing. All 122 separately checked training BN statistics passed. [Phase 3.5 run and tensor archives](https://www.kaggle.com/code/agastyavar/indic-edge-ocr-phase35/versions/3).

| MKLDNN | Torch threads | Exact-input replays | Natural execution | Decoder with exact memory |
|---|---:|---:|---:|---:|
| On | 2 | 870 / 878 | 400 / 878 | 89 / 94 |
| Off | 2 | 870 / 878 | 417 / 878 | 89 / 94 |
| On | 1 | 870 / 878 | 389 / 878 | 94 / 94 |

### Stem and backbone

The first nonzero difference appears at the first stem convolution (training max 1.580e-6). Its isolated replay passes, and the FP64 accumulation diagnostic is closer at 9.537e-7. The first natural tolerance failure is training `stem2a.bn`, with 2 of 491,520 outliers and max 2.670e-5. The same BN on the Paddle convolution result passes every element with max 1.907e-5. Fixed-stat inference BN checks also pass. Training checks use the population variance and check state updates separately.

The first composed layer to fail replay is training `stem2b`, with 5 of 983,040 outliers and max 1.907e-5. Its convolution, BN, and ReLU each pass on their own exact inputs. For the training even-SAME `stem2b.conv`, disabling MKLDNN changes the isolated max from 9.537e-6 to zero, while the upstream-induced natural error remains 1.907e-5. This establishes backend-dependent convolution arithmetic and accumulated error inside the composed stem. The concat, RepDWConv identity and pre-BN sums, SE, channel mixers, and residual additions appear in the machine-readable trace. Diagnostic stem Conv+BN folding helped some layers and hurt others. For example, inference stem1 improves from 3.338e-6 to 1.907e-6, while stem4 worsens from 1.401e-6 to 7.033e-6.

### NRTR projection and decoder

With natural input, `head.before_gtc` has max error 1.841e-4 and 128 outliers. With the Paddle pooled feature map, the transpose boundary and the 768→512 GEMM output both match exactly. All 441 checkpoint tensors were rechecked by the baseline comparator, including the transposed projection weight. The FP64 GEMM diagnostic differs from Paddle by 1.144e-5 and passes all elements. The observed natural projection error is inherited from the backbone.

| Default training boundary | Exact-input max | Natural max |
|---|---:|---:|
| Block 0 self-attention residual LayerNorm | 5.960e-8 | 5.960e-8 |
| Block 0 cross-attention KV | 4.768e-7 | 5.901e-5 |
| Block 0 cross-attention QK matmul | 0 | 1.031e-5 |
| Block 0 cross-attention softmax | 5.588e-9 | 5.215e-8 |
| Block 0 cross-attention output | 4.768e-7 | 2.885e-5 |
| Block 0 cross-attention residual LayerNorm | 4.768e-7 | 1.746e-4 |
| Block 0 final LayerNorm | 7.153e-7 | 1.419e-3 |
| Block 3 output | 2.229e-5 | 4.269e-3 |
| NRTR logits | 1.156e-5 | 3.999e-3 |

The first decoder boundary to materially increase natural error is block 0 cross-attention KV. Its exact-input replay passes. The next two residual LayerNorms amplify natural differences and pass on exact inputs. With the same memory and tokens throughout the decoder, all final logits pass on this worker. The default two-thread run still has five intermediate misses, including block 2 cross-attention output (max 3.166e-4) and block 3 output (max 2.065e-5). One thread removes all five. Block 2's isolated QK and attention-value matmuls are exact, and its softmax differs by at most 1.192e-7. This supports accumulated arithmetic differences, including decoder-local propagation, rather than incorrect weights, masks, or operation order.

### Remaining exact-input failures

| Mode | Boundary | Max absolute error | Outliers |
|---|---|---:|---:|
| Eval | `backbone.blocks2.0` | 2.134e-5 | 2 |
| Eval | `backbone.blocks2` | 2.134e-5 | 2 |
| Eval | `backbone` | 2.587e-5 | 1 |
| Train | `backbone.conv1.stem2b` | 1.907e-5 | 5 |
| Train | `backbone.conv1` | 2.474e-5 | 4 |
| Train | `backbone` | 5.703e-5 | 147 |
| Train | Standalone CTC-logit log-softmax diagnostic | 3.405e-4 | 54 |
| Train | Standalone NRTR log-softmax | 1.469e-4 | 21 |

The repeated block and stage-2 rows describe the same output. The other backbone rows contain multiple operations. The standalone log-softmax failures persist with MKLDNN off and one thread. Paddle's CPU kernel shifts by the input maximum, clips at −64 and evaluates an Eigen exp/sum/log reduction. Neither real logit tensor reaches the clip threshold (minimum shifted logits −46.37 and −18.62). The residual difference is consistent with different reduction implementations. The CTC log-softmax probe is a diagnostic because upstream CTC normalizes inside Warp-CTC. Existing CTC, NRTR and total loss checks still pass. No model change was justified by these experiments. Strict end-to-end FP32 parity remains open.

| New artifact | SHA-256 |
|---|---|
| `paddle-boundaries/boundaries.npz` | `7f9250be5bc89fce92bd93a3242e71a6c3ed924c6d6084ec387d5a660f473615` |
| `replay-on-2/torch_replay.npz` | `9656790b9ec17dd25bc2bd74c7672ccaf6037ce1073df8ee1c2f0433f7fdca12` |
| Default `comparisons.json` | `3477b2b74bbaa1a0b66b6141ede8aea7a77116fe2b7583777e3a594038479104` |
| MKLDNN off `comparisons.json` | `b952e9524a69f69cbc7211bd00393acd5073f304c9bfb28128e1868f0a342cfa` |
| One thread `comparisons.json` | `0e6d6f1d6a927f7da4ea953e37f9659771e77b495e2b0f45d9db3e54c620ea5d` |

## Reproduction

Use Python 3.12.10, Paddle 3.2.0, PyTorch 2.9.1+cpu, NumPy 1.26.4 and OpenCV 4.11.0. The staged job uses the original notebook output by version, reproduces the split checksum, and checks the archive and checkpoint hashes before execution. T4 host selection requests the same worker pool as the original run. All model operations run on CPU.

```bash
python tools/prepare_kaggle_phase35.py --owner agastyavar \
  --directory artifacts/phase35/job-t4 --audit artifacts/hiertext/subset/audit.json \
  --reference agastyavar/indic-edge-ocr-phase23/4 --t4-host
kaggle kernels push -p artifacts/phase35/job-t4 --accelerator NvidiaTeslaT4 --timeout 2600
kaggle kernels output agastyavar/indic-edge-ocr-phase35 -p artifacts/phase35/download
```

For direct execution from the repository, with both frameworks installed in an isolated CPU environment:

```bash
export PYTHONPATH=src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 FLAGS_use_mkldnn=0
inputs=(--archive artifacts/phase23-kaggle-v4/paddle-real-cpu/real_parity.npz
        --original weights/original/PP-OCRv6_medium_rec_pretrained.pdparams
        --checkpoint weights/converted/PP-OCRv6_medium_rec.safetensors
        --dataset artifacts/hiertext/subset)
python tools/trace_parity.py capture "${inputs[@]}" \
  --source references/PaddleOCR --output artifacts/phase35/paddle-boundaries
python tools/trace_parity.py replay "${inputs[@]}" \
  --boundaries artifacts/phase35/paddle-boundaries --output artifacts/phase35/replay-default
python tools/trace_parity.py replay "${inputs[@]}" \
  --boundaries artifacts/phase35/paddle-boundaries --output artifacts/phase35/replay-no-mkldnn \
  --mkldnn off --metrics-only
python tools/trace_parity.py replay "${inputs[@]}" \
  --boundaries artifacts/phase35/paddle-boundaries --output artifacts/phase35/replay-one-thread \
  --threads 1 --metrics-only
```

The boundary manifest maps each module invocation and functional boundary to positional and keyword input arrays and its Paddle output. The primary replay writes corresponding native outputs and per-boundary JSON with finite error percentiles, shape, dtype, relative L2, cosine similarity and pass counts. Natural execution and exact-input replay appear in separate fields. The NRTR decoder also runs with the original Paddle memory tensor and teacher-forced labels. FP64 accumulation and stem BN fusion are diagnostics only.
