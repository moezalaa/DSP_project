"""Fourier transforms written from scratch (no numpy.fft inside).

dft / idft : direct O(N^2) definition, used as a reference.
fft / ifft : recursive Cooley-Tukey radix-2 (decimation in time).
             Works for ANY length: the recursion halves the length while it is
             even; pieces of length <= 64 are finished with a small DFT matrix,
             and a long odd piece is handled by Bluestein's chirp-z algorithm.
fft2/ifft2 : 2D transform = FFT on every row, then on every column.
"""
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=64)
def _dft_matrix(N: int, inverse: bool = False) -> np.ndarray:
    n = np.arange(N)
    sign = 1.0 if inverse else -1.0
    return np.exp(sign * 2j * np.pi * ((np.outer(n, n) % N) / N))


@lru_cache(maxsize=64)
def _twiddle(N: int) -> np.ndarray:
    """exp(-j*2*pi*k/N) for k = 0 .. N/2-1."""
    return np.exp(-2j * np.pi * np.arange(N // 2) / N)


def dft(x: np.ndarray) -> np.ndarray:
    """Direct DFT: X[k] = sum_n x[n] * exp(-j*2*pi*k*n/N).  O(N^2)."""
    x = np.asarray(x, dtype=complex)
    if x.ndim != 1:
        raise ValueError("dft expects a 1D array")
    return _dft_matrix(len(x)) @ x


def idft(X: np.ndarray) -> np.ndarray:
    """Inverse DFT. Returns a complex array (take .real for real signals)."""
    X = np.asarray(X, dtype=complex)
    if X.ndim != 1:
        raise ValueError("idft expects a 1D array")
    return (_dft_matrix(len(X), inverse=True) @ X) / len(X)


_BASE = 64               # pieces this short are finished with a small DFT matrix


def _bluestein(x: np.ndarray) -> np.ndarray:
    """Bluestein (chirp-z) FFT for an odd length N, along the last axis.

    Uses k*n = (k^2 + n^2 - (k-n)^2) / 2 to turn the DFT into a convolution,
    which is then computed with power-of-two FFTs. Cost is O(N log N), so even
    a prime length like 10007 stays fast and needs no N x N matrix.
    """
    N = x.shape[-1]
    n = np.arange(N)
    w = np.exp(-1j * np.pi * ((n * n) % (2 * N)) / N)       # chirp (mod 2N keeps precision)
    M = 1 << int(np.ceil(np.log2(2 * N - 1)))               # power of two >= 2N-1
    a = np.zeros(x.shape[:-1] + (M,), dtype=complex)
    a[..., :N] = x * w
    b = np.zeros(M, dtype=complex)
    b[:N] = np.conj(w)
    b[M - N + 1:] = np.conj(w[1:])[::-1]
    conv = np.conj(_fft_last_axis(np.conj(_fft_last_axis(a) * _fft_last_axis(b)))) / M
    return conv[..., :N] * w


def _fft_last_axis(x: np.ndarray) -> np.ndarray:
    """Cooley-Tukey FFT along the LAST axis. Works on batches (all rows at once)."""
    N = x.shape[-1]
    if N <= _BASE:                      # short piece: plain DFT matrix (symmetric)
        return x @ _dft_matrix(N)
    if N % 2 == 1:                      # long odd piece: cannot split -> Bluestein
        return _bluestein(x)
    even = _fft_last_axis(x[..., 0::2])
    odd = _fft_last_axis(x[..., 1::2])
    t = _twiddle(N) * odd
    return np.concatenate([even + t, even - t], axis=-1)


def fft(x: np.ndarray, n: int = None) -> np.ndarray:
    """FFT of a 1D signal. Output length = len(x), or n if given (zero-pad / cut)."""
    x = np.asarray(x, dtype=complex)
    if x.ndim != 1:
        raise ValueError("fft expects a 1D array (use fft2 for images)")
    if len(x) == 0:
        raise ValueError("fft of an empty signal")
    if n is not None:
        x = np.pad(x[:n], (0, max(0, n - len(x))))
    return _fft_last_axis(x)


def ifft(X: np.ndarray, n: int = None) -> np.ndarray:
    """Inverse FFT using the conjugate trick: ifft(X) = conj(fft(conj(X))) / N."""
    X = np.asarray(X, dtype=complex)
    if X.ndim != 1:
        raise ValueError("ifft expects a 1D array (use ifft2 for images)")
    if n is not None:
        X = np.pad(X[:n], (0, max(0, n - len(X))))
    return np.conj(_fft_last_axis(np.conj(X))) / len(X)


def fft2(img: np.ndarray) -> np.ndarray:
    """2D FFT (rows, then columns). NOT shifted: use utils.fftshift to center it."""
    a = np.asarray(img, dtype=complex)
    if a.ndim != 2:
        raise ValueError("fft2 expects a 2D array")
    a = _fft_last_axis(a)                                   # every row
    a = _fft_last_axis(a.swapaxes(0, 1)).swapaxes(0, 1)     # every column
    return a


def ifft2(F: np.ndarray) -> np.ndarray:
    """2D inverse FFT. Returns a complex array (take .real for images)."""
    a = np.asarray(F, dtype=complex)
    if a.ndim != 2:
        raise ValueError("ifft2 expects a 2D array")
    return np.conj(fft2(np.conj(a))) / a.size
