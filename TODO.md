# PP-OCRv6 medium recognition

- [x] Pin the official PaddleOCR checkout and trace the configured recognition path.
- [x] Verify the Phase 0 module inventory and parameter counts against Paddle.
- [x] Run the Phase 1 checkpoint, inference, activation and loss checks on Kaggle.
- [x] Record artifact provenance and the verification results.
- [x] Implement the native recognition graph, losses, and fixed-dictionary preprocessing.
- [x] Convert and compare all 441 official checkpoint tensors.
- [x] Verify HierText provenance, official partition isolation, and smoke-manifest checksums.
- [x] Verify real-data Accelerate training and checkpoint resume on CPU.
- [ ] Close the remaining FP32 activation and logit parity failures before marking Phases 2/3 complete.
- [x] Verify the FP16 Kaggle training smoke test and checkpoint resume on one T4.
- [x] Reproduce the v4 FP32 baseline and isolate stem, backbone and NRTR error propagation with paired replays.
- [x] Stage Phase 3.6 paired CPU/CUDA forward and backward experiments and an audited 1,080-crop HierText validation subset.
- [x] Run paired FP32 CPU and T4 comparisons and localize the large CPU selected-gradient divergence to Paddle's LayerNorm backward.
- [x] Compare both frozen checkpoints on 1,080 official HierText validation crops on a T4.
