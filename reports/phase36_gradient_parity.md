# Phase 3.6 gradient parity

Each framework starts with the verified original state and consumes the immutable Phase 3.5 training batch. The live forward uses training BatchNorm and eval-mode dropout. Each framework loads the initial parameters and BN buffers again before further independent forward experiments. Training capture records updated running means and variances. CPU and CUDA use their corresponding running-variance behavior from the existing port.

The CTC and NRTR loss experiments differentiate each loss separately from the shared training forward, recording head inputs, logits and selected weights. The full experiment differentiates the sum into the image and selected stem, backbone, projection and decoder parameters. A separate fixed-logit experiment feeds actual Paddle logits to both loss operators so inherited backbone error cannot affect that comparison. Converted Linear weight gradients use Torch layout in the metrics. No optimizer update is part of these runs.

The baseline uses the existing native PyTorch loss. Ablations change CTC alone, NRTR alone, then NRTR's reduction order. The original activation thresholds apply to gradient rows. Reports include the element count and tolerance pass count, maximum and mean absolute error, relative L2, cosine and shape. Reproduction uses the paired commands in [Numerical equivalence](phase36_numerical_equivalence.md).
