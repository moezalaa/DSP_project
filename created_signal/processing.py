"""Created Signal: generation, noise, filtering and analysis (pure functions, no GUI).

Angles are in RADIANS here (the GUI converts from degrees).
All filtering is done with our own core/ code (core.convolution, core.filters,
core.fourier), never with numpy/scipy.
"""
import math

import numpy as np

from core.convolution import convolve1d
from core.filters import freq_filter_1d
from core.fourier import fft, ifft
from core.utils import freq_axis, magnitude, phase, one_sided

WAVE_KINDS = ("sine", "cosine", "square", "triangle", "sawtooth")
CONV_KINDS = ("moving_average", "difference", "gaussian", "custom")
FREQ_KINDS = ("lowpass", "highpass", "bandpass", "notch")
FILTER_KINDS = ("none",) + CONV_KINDS + ("first_order",) + FREQ_KINDS


# ----------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------
def generate_signal(kind, amp, freq, phase_rad, fs, duration):
    """Generate one waveform.  Returns (t, x).

    kind     : 'sine' | 'cosine' | 'square' | 'triangle' | 'sawtooth'
    amp      : peak amplitude
    freq     : frequency in Hz
    phase_rad: phase offset in radians
    fs       : sampling rate in Hz
    duration : length in seconds
    """
    if kind not in WAVE_KINDS:
        raise ValueError(f"kind must be one of {WAVE_KINDS}")
    if fs <= 0:
        raise ValueError("Sampling rate must be positive")
    if duration <= 0:
        raise ValueError("Duration must be positive")
    if freq < 0:
        raise ValueError("Frequency cannot be negative")
    n = int(round(fs * duration))
    if n < 2:
        raise ValueError("Duration is too short: need at least 2 samples")

    t = np.arange(n) / fs
    theta = 2 * np.pi * freq * t + phase_rad
    if kind == "sine":
        x = np.sin(theta)
    elif kind == "cosine":
        x = np.cos(theta)
    elif kind == "square":
        x = np.where(np.mod(theta, 2 * np.pi) < np.pi, 1.0, -1.0)
    elif kind == "triangle":
        x = (2 / np.pi) * np.arcsin(np.sin(theta))
    else:  # sawtooth: ramps from -1 to +1 each period
        x = np.mod(theta, 2 * np.pi) / np.pi - 1.0
    return t, amp * x


def combine_signals(signals):
    """Sum several signals of the same length into one."""
    if len(signals) == 0:
        raise ValueError("Enable at least one signal component")
    signals = [np.asarray(s, dtype=float) for s in signals]
    if any(s.shape != signals[0].shape for s in signals):
        raise ValueError("All signals must have the same length")
    return np.sum(signals, axis=0)


def add_noise(x, snr_db, seed=None):
    """Add white Gaussian noise so that the signal-to-noise ratio equals snr_db."""
    x = np.asarray(x, dtype=float)
    power = np.mean(x ** 2)
    if power == 0:
        raise ValueError("Cannot set an SNR on an all-zero signal")
    noise_power = power / (10 ** (snr_db / 10))
    rng = np.random.default_rng(seed)
    return x + rng.normal(0.0, np.sqrt(noise_power), size=x.shape)


# ----------------------------------------------------------------------------
# Filtering
# ----------------------------------------------------------------------------
def make_kernel(kernel_type, size=5, kernel=None):
    """Build the impulse response used by apply_conv_filter."""
    if kernel_type == "moving_average":
        if size < 1:
            raise ValueError("Kernel size must be at least 1")
        return np.ones(size) / size
    if kernel_type == "difference":
        return np.array([1.0, -1.0])
    if kernel_type == "gaussian":
        if size < 1:
            raise ValueError("Kernel size must be at least 1")
        sigma = max(size / 6.0, 1e-3)
        k = np.arange(size) - (size - 1) / 2
        g = np.exp(-(k ** 2) / (2 * sigma ** 2))
        return g / g.sum()
    if kernel_type == "custom":
        if kernel is None or len(kernel) == 0:
            raise ValueError("Custom kernel is empty")
        return np.asarray(kernel, dtype=float)
    raise ValueError(f"kernel_type must be one of {CONV_KINDS}")


def apply_conv_filter(x, kernel_type, size=5, kernel=None):
    """Filter in the time domain with convolve1d (output has the same length as x)."""
    h = make_kernel(kernel_type, size, kernel)
    x = np.asarray(x, dtype=float)
    return convolve1d(x, h, mode="same")[:len(x)]


def apply_freq_filter(x, fs, filter_type, cutoff, order=None, width=2.0):
    """Frequency-domain LPF / HPF / BPF / Notch (wraps core.filters.freq_filter_1d)."""
    y, _, _ = freq_filter_1d(x, fs, filter_type, cutoff, order, width)
    return y


def apply_filter_spec(x, fs, spec):
    """Apply the filter described by `spec` (a dict). Used by the GUI.

    spec["type"] is one of FILTER_KINDS. Other keys:
      conv filters : size, kernel (list, for 'custom')
      freq filters : cutoff, cutoff_high (bandpass), order (None/0 = ideal), width (notch)
    """
    kind = spec.get("type", "none")
    if kind == "none":
        return np.asarray(x, dtype=float).copy()
    if kind in CONV_KINDS:
        return apply_conv_filter(x, kind, int(spec.get("size", 5)), spec.get("kernel"))
    if kind == "first_order":
        return first_order_difference_equation(x, spec.get("a", 0.9), spec.get("b", 0.1))
    if kind in FREQ_KINDS:
        order = spec.get("order") or None          # 0 / None -> ideal
        cutoff = (spec["cutoff"], spec["cutoff_high"]) if kind == "bandpass" else spec["cutoff"]
        return apply_freq_filter(x, fs, kind, cutoff, order, spec.get("width", 2.0))
    raise ValueError(f"Unknown filter type: {kind}")


def first_order_difference_equation(x, a, b):
    """First-order difference equation  y[n] = a*y[n-1] + b*x[n]  (initial rest, y[-1] = 0).

    It is a recursive (IIR) LTI system: impulse response h[n] = b * a^n for n >= 0,
    stable when |a| < 1.  Written with an explicit loop on purpose.
    """
    x = np.asarray(x, dtype=float)
    y = np.zeros(len(x))
    previous = 0.0
    for n in range(len(x)):
        previous = a * previous + b * x[n]
        y[n] = previous
    return y


def is_stable(a):
    """The system y[n] = a*y[n-1] + b*x[n] is stable only when |a| < 1."""
    return abs(a) < 1


def convolution_vs_frequency(x, h):
    """Linear convolution in time vs multiplication in frequency, with and without zero-padding.

    direct   : convolve1d(x, h) 'full', length N+M-1  (the true linear convolution)
    padded   : ifft(fft(x, L) * fft(h, L)) with L = N+M-1   -> equals `direct`
    circular : the same but WITHOUT zero-padding (length max(N, M)) -> circular convolution:
               the tail of the result wraps around onto the first M-1 samples.
    """
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    N, M = len(x), len(h)
    L = N + M - 1
    direct = convolve1d(x, h, mode="full")
    padded = ifft(fft(x, n=L) * fft(h, n=L)).real
    Nc = max(N, M)
    circular = ifft(fft(x, n=Nc) * fft(h, n=Nc)).real
    return {"direct": direct, "padded": padded, "circular": circular,
            "err_padded": padded - direct, "err_circular": circular - direct[:Nc]}


# ----------------------------------------------------------------------------
# DTFT from its defining equation (explicit loops)
# ----------------------------------------------------------------------------
MAX_DTFT_TERMS = 1_500_000        # samples x frequency points: keeps the loops quick


def dtft(x, omegas):
    """DTFT from the definition  X(w) = sum_n x[n] * exp(-j*w*n)  using plain loops.

    x      : the (short) signal, n = 0 .. N-1
    omegas : angular frequencies in rad/sample
    Returns the complex values X(w) for every w. No FFT is used.
    """
    xs = [float(v) for v in x]
    omegas = [float(w) for w in omegas]
    if len(xs) * len(omegas) > MAX_DTFT_TERMS:
        raise ValueError("DTFT is too large for explicit loops: use fewer samples or frequency points")
    X = np.zeros(len(omegas), dtype=complex)
    for i, w in enumerate(omegas):               # loop over frequency points
        re = 0.0
        im = 0.0
        for n, xn in enumerate(xs):              # loop over samples
            re += xn * math.cos(w * n)
            im -= xn * math.sin(w * n)
        X[i] = complex(re, im)
    return X


def dtft_spectrum(x, n_samples, n_points):
    """DTFT of the first n_samples of x on n_points frequencies in [-pi, pi].

    Returns (omegas, X, magnitude, phase), angles in radians.
    """
    x = np.asarray(x, dtype=float)
    if n_samples < 2 or n_points < 2:
        raise ValueError("DTFT needs at least 2 samples and 2 frequency points")
    xs = x[:int(n_samples)]
    omegas = np.linspace(-np.pi, np.pi, int(n_points))
    X = dtft(xs, omegas)
    return omegas, X, magnitude(X), phase(X)


# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------
def analyze(x, fs):
    """One-sided spectrum of x.  Returns (freqs, magnitude, phase).

    magnitude is an AMPLITUDE spectrum: a sine of amplitude A shows a peak of A.
    phase is in radians; bins whose magnitude is negligible get phase 0
    (their phase is just numerical noise and would clutter the plot).
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    X = fft(x)
    freqs, X = one_sided(freq_axis(N, fs), X)
    mag = magnitude(X) / N
    mag[1:] *= 2
    if N % 2 == 0:
        mag[-1] /= 2                               # Nyquist bin is not doubled
    ph = phase(X)
    ph[mag < 1e-9 * max(mag.max(), 1e-300)] = 0.0
    return freqs, mag, ph


def nyquist_warnings(components, fs):
    """List of messages for components above the Nyquist frequency (aliasing)."""
    msgs = []
    for i, c in enumerate(components, start=1):
        if c["freq"] > fs / 2:
            msgs.append(f"Component {i}: {c['freq']:g} Hz is above Nyquist "
                        f"({fs / 2:g} Hz), so it will alias")
    return msgs


# ----------------------------------------------------------------------------
# Whole pipeline (what the GUI calls)
# ----------------------------------------------------------------------------
def run_pipeline(params):
    """Generate -> combine -> (noise) -> filter -> analyze.

    params = {
      "fs": 1000, "duration": 1.0,
      "components": [{"kind": "sine", "amp": 1, "freq": 5, "phase": 0.0}, ...],
      "noise_snr": None or float (dB), "seed": int,
      "filter": {"type": "lowpass", "cutoff": 20, ...},
    }
    Returns a dict: t, clean, input, output, freqs, mag_in, ph_in, mag_out, ph_out.
    """
    fs, duration = params["fs"], params["duration"]
    parts = []
    t = None
    for c in params["components"]:
        t, x = generate_signal(c["kind"], c["amp"], c["freq"], c["phase"], fs, duration)
        parts.append(x)
    clean = combine_signals(parts)

    snr = params.get("noise_snr")
    noisy = add_noise(clean, snr, params.get("seed")) if snr is not None else clean
    out = apply_filter_spec(noisy, fs, params.get("filter", {"type": "none"}))

    freqs, mag_in, ph_in = analyze(noisy, fs)
    _, mag_out, ph_out = analyze(out, fs)
    return {"t": t, "clean": clean, "input": noisy, "output": out, "freqs": freqs,
            "mag_in": mag_in, "ph_in": ph_in, "mag_out": mag_out, "ph_out": ph_out}
