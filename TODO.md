# PP-OCRv6 medium recognition

- [x] Pin the official PaddleOCR checkout and trace the configured recognition path.
- [x] Verify the Phase 0 module inventory and parameter counts against Paddle.
- [x] Run the Phase 1 checkpoint, inference, activation and loss checks on Kaggle.
- [x] Record artifact provenance and the verification results.
- [ ] Start Phase 2 after the Paddle reference passes.
- [ ] For the PyTorch training smoke test, use Hugging Face Accelerate with configurable mixed precision. Use FP16 on Kaggle T4 and retain an FP32 configuration for controlled checks.
- [ ] Cache downloads and deterministic dataset inputs on Kaggle. Apply random augmentations after the cache and keep FP32 parity runs independent of training speed settings.
