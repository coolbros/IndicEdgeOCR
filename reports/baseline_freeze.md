# Frozen PP-OCRv6 medium baseline

Use the official converted PP-OCRv6_medium_rec checkpoint as the frozen Phase 4 baseline. The converter applies an explicit same-name mapping, canonicalizes Paddle BN buffers and transposes 39 Linear weights. The default model loads all 441 tensors exactly and has 55,527,798 trainable parameters. Default CTC has 18,710 classes and blank 0. Default NRTR has 18,714 classes and PAD/BOS/EOS 0/2/3. PaddleOCR source is pinned at `dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf`.

| Immutable artifact | SHA-256 |
|---|---|
| Official Paddle checkpoint | `c4a110423c23739f157ff870e27f41649d8c89e4c8df5230c2cd2e9fd90a7e92` |
| Converted default PyTorch checkpoint | `0afc351e31cea4f6d670e68652bd22a11ab44273584e156159dbdb83fafb02ef` |
| Regression validation split manifest | `a6c78abb1c0e6e5f2e3c3c814fd9daf1bc582d35bc3819c091a91e61a7df5ed7` |

The [fresh T4 inference comparison](https://www.kaggle.com/code/agastyavar/indic-edge-ocr-phase38-inference/versions/1) matched all 1,080 decoded strings from the official HierText validation subset. Both CER and NFC CER are 12.2747%, with WER 21.3889%. Confidence mean and max absolute differences are 2.705e-7 and 1.788e-5. Paddle and Torch processed 224.62 and 220.62 crops per second respectively, with median synchronized batch latencies of 57.23 and 60.50 ms. Both consumed identical preprocessed batch bytes in width-bucketed batches, with up to 16 crops per batch. Timing excludes decode and input transfer. Variability across batch sizes for the entire set was not measured in this run.

- Architectural and checkpoint evidence is complete for the official configuration. See [baseline audit](phase38_baseline_audit.md).
- Inference equivalence is verified on the audited T4 fixture. Phase 3.5–3.7 also supply CPU evidence for small tensor and decoding fixtures. This phase's newly measured 1,080-crop rerun is CUDA.
- Numerical forward parity retains strict outliers on natural training forwards, despite exact inference string agreement on the measured fixture.
- Full-graph backward still has strict FP32 outliers. See [backward attribution](phase38_backward_parity.md).
- Training uses separate experimental checkpoint files. See [training validation](phase38_training_validation.md).
- The validation subset is predominantly ASCII, with 36 non-ASCII examples. It does not establish Indic accuracy or training-trajectory equivalence.

The conversion and inference commands are documented in the [baseline audit and validation instructions](phase38_training_validation.md).

The reproduced T4 recognition summary is SHA-256 `42d3006e50c3dd0d91481fb0389afdd4a3df74292a93f03b79924f0492de9c69`.

A local Windows CPU replay of the first immutable Paddle inference crop used the configured default heads. Decoded strings match at batch sizes 1 and 4 and match the archived Paddle string. The CTC probability maximum changes by 1.192e-7 from batching and differs from Paddle by at most 1.257e-7 at batch size 1. This is a single-crop batch sensitivity check.

The operational baseline is frozen for Phase 4. The validation run completed 101 nonzero-LR updates and a checked resume. The remaining strict CPU and CUDA gradient outliers are documented with their measured practical scope. Full-graph bitwise or strict FP32 parity is not a claim of this freeze. The next phase starts with licensing and split verification for an Indic evaluation corpus and offline tokenization comparisons.
