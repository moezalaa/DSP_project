"""Frequency-domain filters for 1D signals and 2D images (built on core.fourier).

1D : freq_filter_1d   -> lowpass / highpass / bandpass / notch
2D : freq_mask_2d     -> Ideal / Gaussian / Butterworth masks (centered)
     freq_filter_2d   -> applies a mask to an image
apply_filter(...)     -> one entry point for the GUI: returns (output, magnitude, phase)
"""
import numpy as np

from core.fourier import fft, ifft, fft2, ifft2
from core.utils import magnitude, phase, fftshift, ifftshift, to_db

FILTER_TYPES_1D = ("lowpass", "highpass", "bandpass", "notch")
FILTER_TYPES_2D = ("lowpass", "highpass")
KINDS_2D = ("ideal", "gaussian", "butterworth")


# ----------------------------------------------------------------------------
# 1D
# ----------------------------------------------------------------------------
def _abs_freqs(N: int, fs: float) -> np.ndarray:
    """|frequency| in Hz of each FFT bin (bins above N/2 are the negative freqs)."""
    k = np.arange(N)
    return np.minimum(k, N - k) * fs / N


_EPS = 1e-9      # relative tolerance so a bin exactly AT the cutoff is not lost to rounding


def _lowpass_response(f, cutoff, order):
    if order is None:                       # ideal (brick wall)
        return (f <= cutoff * (1 + _EPS)).astype(float)
    return 1.0 / (1.0 + (f / cutoff) ** (2 * order))     # Butterworth


def freq_filter_1d(x, fs, filter_type, cutoff, order=None, width=2.0):
    """Filter a 1D signal in the frequency domain.

    filter_type: 'lowpass' | 'highpass' | 'bandpass' | 'notch'
    cutoff     : Hz (float). For 'bandpass' use a (low, high) tuple.
    order      : None -> ideal brick-wall filter, int -> Butterworth of that order.
    width      : notch width in Hz (total stop band = cutoff +/- width/2).

    Returns (y, X, Y): filtered signal, input spectrum, output spectrum.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 2:
        raise ValueError("Signal must be a 1D array with at least 2 samples")
    if fs <= 0:
        raise ValueError("Sampling rate must be positive")
    if filter_type not in FILTER_TYPES_1D:
        raise ValueError(f"filter_type must be one of {FILTER_TYPES_1D}")

    N = len(x)
    f = _abs_freqs(N, fs)

    if filter_type == "bandpass":
        try:
            lo, hi = cutoff
        except (TypeError, ValueError):
            raise ValueError("bandpass needs cutoff=(low, high)")
        if not 0 <= lo < hi:
            raise ValueError("bandpass needs 0 <= low < high")
        H = (f >= lo * (1 - _EPS)) & (f <= hi * (1 + _EPS)) if order is None else \
            _lowpass_response(f, hi, order) * (1 - _lowpass_response(f, max(lo, 1e-9), order))
        H = H.astype(float)
    else:
        if np.ndim(cutoff) != 0 or cutoff <= 0:
            raise ValueError("cutoff must be a positive number")
        if filter_type == "lowpass":
            H = _lowpass_response(f, cutoff, order)
        elif filter_type == "highpass":
            H = 1.0 - _lowpass_response(f, cutoff, order)
        else:  # notch
            H = 1.0 - (np.abs(f - cutoff) <= width / 2 + _EPS).astype(float)

    X = fft(x)
    Y = X * H
    y = ifft(Y).real
    return y, X, Y


# ----------------------------------------------------------------------------
# 2D
# ----------------------------------------------------------------------------
def freq_mask_2d(shape, filter_type, cutoff, kind="ideal", order=2):
    """Real mask in [0, 1] for a CENTERED (fftshift-ed) 2D spectrum.

    shape      : (rows, cols) of the image
    filter_type: 'lowpass' | 'highpass'
    cutoff     : radius D0 in pixels from the center of the spectrum
    kind       : 'ideal' | 'gaussian' | 'butterworth'
    order      : Butterworth order n
    """
    if filter_type not in FILTER_TYPES_2D:
        raise ValueError(f"filter_type must be one of {FILTER_TYPES_2D}")
    if kind not in KINDS_2D:
        raise ValueError(f"kind must be one of {KINDS_2D}")
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")

    rows, cols = shape
    u = np.arange(rows) - rows // 2
    v = np.arange(cols) - cols // 2
    D = np.sqrt(u[:, None] ** 2 + v[None, :] ** 2)

    if kind == "ideal":
        low = (D <= cutoff).astype(float)
    elif kind == "gaussian":
        low = np.exp(-(D ** 2) / (2.0 * cutoff ** 2))
    else:
        low = 1.0 / (1.0 + (D / cutoff) ** (2 * order))

    return low if filter_type == "lowpass" else 1.0 - low


def freq_filter_2d(img, filter_type, cutoff, kind="ideal", order=2):
    """Filter an image in the frequency domain.

    Returns (out, F_shifted, G_shifted): output image (real, same shape),
    centered spectrum of the input, centered spectrum of the output.
    """
    img = np.asarray(img, dtype=float)
    if img.ndim != 2:
        raise ValueError("Image must be a 2D (grayscale) array")
    F = fftshift(fft2(img))
    G = F * freq_mask_2d(img.shape, filter_type, cutoff, kind, order)
    out = ifft2(ifftshift(G)).real
    return out, F, G


# ----------------------------------------------------------------------------
# One entry point for the GUI
# ----------------------------------------------------------------------------
def apply_filter(signal, filter_type, **params):
    """Apply a frequency-domain filter and return (output, magnitude, phase).

    magnitude / phase are those of the OUTPUT's spectrum.

    1D signal : params = fs, cutoff, [order], [width]
                magnitude/phase have length N, in raw FFT order.
    2D image  : params = cutoff, [kind='ideal'], [order=2]
                magnitude is log-scaled (dB-like) and centered, phase is centered.
    """
    signal = np.asarray(signal, dtype=float)
    if signal.ndim == 1:
        y, _, Y = freq_filter_1d(signal, params["fs"], filter_type, params["cutoff"],
                                 params.get("order"), params.get("width", 2.0))
        return y, magnitude(Y), phase(Y)
    if signal.ndim == 2:
        out, _, G = freq_filter_2d(signal, filter_type, params["cutoff"],
                                   params.get("kind", "ideal"), params.get("order", 2))
        return out, np.log1p(magnitude(G)), phase(G)
    raise ValueError("signal must be 1D or 2D")
