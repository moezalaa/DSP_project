"""1D and 2D convolution written from scratch (no numpy.convolve / scipy inside).

Both use the "shift and add" form: for every kernel tap we add a scaled,
shifted copy of the input. It is the exact convolution sum, but vectorized,
so it is fast enough for full-size images.
"""
import numpy as np


def convolve1d(x: np.ndarray, h: np.ndarray, mode: str = "full") -> np.ndarray:
    """y[n] = sum_k h[k] * x[n-k].

    mode: 'full'  -> length N+M-1
          'same'  -> length max(N, M), centered (same rule as numpy.convolve)
          'valid' -> only where the signals fully overlap
    """
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    if x.ndim != 1 or h.ndim != 1:
        raise ValueError("convolve1d expects 1D arrays")
    if len(x) == 0 or len(h) == 0:
        raise ValueError("convolve1d: empty input")
    if mode not in ("full", "same", "valid"):
        raise ValueError("mode must be 'full', 'same' or 'valid'")

    N, M = len(x), len(h)
    full = np.zeros(N + M - 1)
    for k in range(M):
        full[k:k + N] += h[k] * x

    if mode == "full":
        return full
    if mode == "same":
        L = max(N, M)
        start = (len(full) - L) // 2
        return full[start:start + L]
    L = max(N, M) - min(N, M) + 1          # valid
    start = min(N, M) - 1
    return full[start:start + L]


def convolve2d(img: np.ndarray, kernel: np.ndarray, padding: str = "zero") -> np.ndarray:
    """True 2D convolution (kernel is flipped). Output has the SAME shape as img.

    padding: 'zero' | 'reflect' | 'edge'  (how the borders are extended)
    """
    img = np.asarray(img, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    if img.ndim != 2 or kernel.ndim != 2:
        raise ValueError("convolve2d expects 2D arrays")
    modes = {"zero": "constant", "reflect": "reflect", "edge": "edge"}
    if padding not in modes:
        raise ValueError("padding must be 'zero', 'reflect' or 'edge'")

    H, W = img.shape
    kh, kw = kernel.shape
    ch, cw = (kh - 1) // 2, (kw - 1) // 2
    padded = np.pad(img, ((kh - 1 - ch, ch), (kw - 1 - cw, cw)), mode=modes[padding])

    kf = kernel[::-1, ::-1]                # flip -> convolution (not correlation)
    out = np.zeros((H, W))
    for i in range(kh):
        for j in range(kw):
            out += kf[i, j] * padded[i:i + H, j:j + W]
    return out
