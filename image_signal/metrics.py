import numpy as np

PEAK = 255.0


def _pair(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Images must have the same shape, got {a.shape} and {b.shape}")
    if a.size == 0:
        raise ValueError("Images must not be empty")
    return a, b


def mse(a, b):
    """Mean squared error: mean((a - b)^2)."""
    a, b = _pair(a, b)
    return float(np.mean((a - b) ** 2))


def psnr(a, b, peak=PEAK):
    """Peak signal-to-noise ratio in dB: 10*log10(peak^2 / MSE). Identical images give inf."""
    m = mse(a, b)
    return float("inf") if m == 0 else float(10.0 * np.log10(peak ** 2 / m))


def image_metrics(a, b):
    """All metrics at once: dict with mae, mse, rmse, max_error, psnr."""
    a, b = _pair(a, b)
    diff = np.abs(a - b)
    m = float(np.mean(diff ** 2))
    return {"mae": float(np.mean(diff)), "mse": m, "rmse": float(np.sqrt(m)),
            "max_error": float(np.max(diff)),
            "psnr": float("inf") if m == 0 else float(10.0 * np.log10(PEAK ** 2 / m))}


def calculate_metrics(ours, reference):
    """Backward-compatible tuple (mae, rmse, max_error, psnr), used by the comparison script."""
    r = image_metrics(ours, reference)
    return r["mae"], r["rmse"], r["max_error"], r["psnr"]
