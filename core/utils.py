"""Small helpers shared by every module: magnitude, phase, dB, shifting, axes."""
import numpy as np


def magnitude(X: np.ndarray) -> np.ndarray:
    """|X| for a complex spectrum."""
    X = np.asarray(X)
    return np.sqrt(X.real ** 2 + X.imag ** 2)


def phase(X: np.ndarray) -> np.ndarray:
    """Phase angle of X in radians, in [-pi, pi]."""
    X = np.asarray(X)
    return np.arctan2(X.imag, X.real)


def to_db(m: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Convert a magnitude to decibels: 20*log10(m + eps)."""
    return 20.0 * np.log10(np.asarray(m, dtype=float) + eps)


def normalize(x: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Linearly rescale x to the range [lo, hi]. A constant array maps to lo."""
    x = np.asarray(x, dtype=float)
    xmin, xmax = x.min(), x.max()
    if xmax == xmin:
        return np.full_like(x, lo)
    return lo + (x - xmin) * (hi - lo) / (xmax - xmin)


def freq_axis(n: int, fs: float, shift: bool = False) -> np.ndarray:
    """Frequency (Hz) of every bin of an n-point FFT.

    shift=False: [0, fs/n, ..., (n-1)*fs/n]  (raw FFT order)
    shift=True : centered around 0 (use together with fftshift).
    """
    k = np.arange(n)
    if shift:
        k = k - n // 2
    return k * fs / n


def fftshift(X: np.ndarray, axes=None) -> np.ndarray:
    """Move the zero-frequency component to the center (works for 1D and 2D)."""
    X = np.asarray(X)
    axes = tuple(range(X.ndim)) if axes is None else tuple(axes)
    return np.roll(X, [X.shape[a] // 2 for a in axes], axis=axes)


def ifftshift(X: np.ndarray, axes=None) -> np.ndarray:
    """Inverse of fftshift."""
    X = np.asarray(X)
    axes = tuple(range(X.ndim)) if axes is None else tuple(axes)
    return np.roll(X, [-(X.shape[a] // 2) for a in axes], axis=axes)


def one_sided(freqs: np.ndarray, *spectra: np.ndarray):
    """Keep only the non-negative half (bins 0..N/2) of freqs and each spectrum."""
    half = len(freqs) // 2 + 1
    return (freqs[:half],) + tuple(np.asarray(s)[:half] for s in spectra)
