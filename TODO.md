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
