"""Reference implementations of the SAME operations using numpy / scipy.

They are written independently from our own code on purpose (different library
calls), so comparing the two tells us how accurate our from-scratch versions are.
"""
import numpy as np
from scipy import signal as sps

_EPS = 1e-9


def ref_generate(kind, amp, freq, phase_rad, fs, duration):
    """Waveform with scipy.signal (square / sawtooth) and numpy. Returns (t, x)."""
    n = int(round(fs * duration))
    t = np.arange(n) / fs
    theta = 2 * np.pi * freq * t + phase_rad
    if kind == "sine":
        x = np.sin(theta)
    elif kind == "cosine":
        x = np.cos(theta)
    elif kind == "square":
        x = sps.square(theta)
    elif kind == "triangle":
        x = sps.sawtooth(theta + np.pi / 2, width=0.5)
    elif kind == "sawtooth":
        x = sps.sawtooth(theta)
    else:
        raise ValueError(f"Unknown kind: {kind}")
    return t, amp * x


def ref_analyze(x, fs):
    """One-sided amplitude spectrum with numpy.fft.rfft. Returns (freqs, mag, phase)."""
    x = np.asarray(x, dtype=float)
    N = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    mag = np.abs(X) / N
    mag[1:] *= 2
    if N % 2 == 0:
        mag[-1] /= 2
    ph = np.angle(X)
    ph[mag < 1e-9 * max(mag.max(), 1e-300)] = 0.0
    return freqs, mag, ph


def ref_roundtrip(x):
    """ifft(fft(x)) with numpy."""
    return np.fft.ifft(np.fft.fft(np.asarray(x, dtype=float))).real


def ref_kernel(kind, size=5, kernel=None):
    if kind == "moving_average":
        return np.full(size, 1.0 / size)
    if kind == "difference":
        return np.array([1.0, -1.0])
    if kind == "gaussian":
        g = sps.windows.gaussian(size, std=max(size / 6.0, 1e-3))
        return g / g.sum()
    if kind == "custom":
        return np.asarray(kernel, dtype=float)
    raise ValueError(f"Unknown kernel: {kind}")


def ref_conv_filter(x, kind, size=5, kernel=None):
    """Time-domain filter with scipy.signal.convolve (direct method), 'same' length."""
    x = np.asarray(x, dtype=float)
    h = ref_kernel(kind, size, kernel)
    return sps.convolve(x, h, mode="same", method="direct")[:len(x)]


def ref_freq_filter(x, fs, kind, cutoff, order=None, width=2.0):
    """Frequency-domain filter with numpy.fft (same mask definition, numpy's FFT)."""
    x = np.asarray(x, dtype=float)
    N = len(x)
    f = np.abs(np.fft.fftfreq(N, d=1.0 / fs))

    def low(c):
        if order is None:
            return (f <= c * (1 + _EPS)).astype(float)
        return 1.0 / (1.0 + (f / c) ** (2 * order))

    if kind == "lowpass":
        H = low(cutoff)
    elif kind == "highpass":
        H = 1.0 - low(cutoff)
    elif kind == "bandpass":
        lo, hi = cutoff
        if order is None:
            H = ((f >= lo * (1 - _EPS)) & (f <= hi * (1 + _EPS))).astype(float)
        else:
            H = low(hi) * (1.0 - low(max(lo, 1e-9)))
    elif kind == "notch":
        H = 1.0 - (np.abs(f - cutoff) <= width / 2 + _EPS).astype(float)
    else:
        raise ValueError(f"Unknown filter: {kind}")
    return np.fft.ifft(np.fft.fft(x) * H).real


def ref_difference_equation(x, a, b):
    """y[n] = a*y[n-1] + b*x[n] with scipy.signal.lfilter."""
    return sps.lfilter([b], [1.0, -a], np.asarray(x, dtype=float))


def ref_dtft_on_dft_grid(x):
    """DTFT sampled at w_k = 2*pi*k/N is exactly the DFT: numpy.fft.fft(x)."""
    return np.fft.fft(np.asarray(x, dtype=float))


def ref_linear_convolution(x, h):
    """numpy.convolve 'full'."""
    return np.convolve(np.asarray(x, dtype=float), np.asarray(h, dtype=float), mode="full")


def ref_apply_filter_spec(x, fs, spec):
    """Same dict interface as processing.apply_filter_spec, but with numpy/scipy."""
    from created_signal.processing import CONV_KINDS, FREQ_KINDS
    kind = spec.get("type", "none")
    if kind == "none":
        return np.asarray(x, dtype=float).copy()
    if kind in CONV_KINDS:
        return ref_conv_filter(x, kind, int(spec.get("size", 5)), spec.get("kernel"))
    if kind == "first_order":
        return ref_difference_equation(x, spec.get("a", 0.9), spec.get("b", 0.1))
    if kind in FREQ_KINDS:
        order = spec.get("order") or None
        cutoff = (spec["cutoff"], spec["cutoff_high"]) if kind == "bandpass" else spec["cutoff"]
        return ref_freq_filter(x, fs, kind, cutoff, order, spec.get("width", 2.0))
    raise ValueError(f"Unknown filter type: {kind}")
