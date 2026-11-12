"""Speech core: audio I/O, spectra, filters, echo and noise reduction.

Owner: Fatma.  Used by: Ghada (speech_signal/gui.py).

Rules (see CONTRACT.md):
  * Audio is a mono float64 numpy array with values in [-1, 1].
  * Nothing here opens a window or plots; functions only return data.
  * Bad input raises ValueError with a clear message (the GUI shows it).

Everything is built on OUR OWN core modules (core.fourier, core.convolution,
core.filters, core.utils), not on numpy.fft / scipy.signal.
"""
import numpy as np

from core.convolution import convolve1d
from core.filters import freq_filter_1d
from core.fourier import fft, ifft
from core.utils import freq_axis, magnitude, one_sided, phase, to_db

EPS = 1e-12


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def _check_signal(x, name="signal"):
    """Return x as a float64 1D array, or raise ValueError if it is unusable."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"{name} must be a 1D (mono) array")
    if len(x) < 2:
        raise ValueError(f"{name} is empty or too short (need at least 2 samples)")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return x


def _check_fs(fs):
    if not np.isscalar(fs) or fs <= 0:
        raise ValueError("Sampling rate fs must be a positive number")


def _hamming(n):
    """Hamming window written by hand: w[k] = 0.54 - 0.46*cos(2*pi*k/(n-1))."""
    if n == 1:
        return np.ones(1)
    return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n) / (n - 1))


def _next_pow2(n):
    return 1 << int(np.ceil(np.log2(max(n, 1))))


def _fft_convolve(x, h):
    """Full linear convolution using the convolution theorem and OUR fft.

    y = IFFT( FFT(x) * FFT(h) )  with both zero-padded to length >= N+M-1.
    Same result as convolve1d(x, h, 'full') but O(N log N): needed because an
    echo impulse response has ~fs*delay taps (tens of thousands).
    """
    L = len(x) + len(h) - 1
    n = _next_pow2(L)
    Y = fft(x, n) * fft(h, n)
    return ifft(Y).real[:L]


# ----------------------------------------------------------------------------
# Audio input / output
# ----------------------------------------------------------------------------
def load_audio(path):
    """Read an audio file. Returns (x, fs): mono float64 in [-1, 1], and fs in Hz.

    Stereo (or more channels) is averaged to mono.
    Raises ValueError if the file is missing, unreadable or empty.
    """
    import soundfile as sf

    try:
        data, fs = sf.read(str(path), dtype="float64", always_2d=True)
    except Exception as e:                       # missing / corrupt / unsupported
        raise ValueError(f"Cannot read audio file '{path}': {e}") from e
    if data.shape[0] < 2:
        raise ValueError(f"Audio file '{path}' is empty")
    x = data.mean(axis=1)                        # stereo -> mono
    if not np.all(np.isfinite(x)):
        raise ValueError(f"Audio file '{path}' contains invalid samples")
    return np.clip(x, -1.0, 1.0), int(fs)


def record_audio(duration, fs=44100):
    """Record `duration` seconds from the default microphone (blocks until done).

    Returns a mono float64 array in [-1, 1].  Call it from a QThread in the GUI.
    Raises ValueError for bad arguments or if no microphone is available.
    """
    if duration <= 0:
        raise ValueError("Recording duration must be positive")
    if duration > 600:
        raise ValueError("Recording duration is limited to 600 seconds")
    _check_fs(fs)
    try:
        import sounddevice as sd

        frames = int(round(duration * fs))
        data = sd.rec(frames, samplerate=int(fs), channels=1, dtype="float64")
        sd.wait()
    except Exception as e:                       # no PortAudio / no mic / busy device
        raise ValueError(f"Could not record from the microphone: {e}") from e
    return np.clip(np.asarray(data, dtype=float).ravel(), -1.0, 1.0)


def save_audio(path, x, fs):
    """Write x (mono, [-1, 1]) to `path` as a 16-bit WAV file. Values outside [-1, 1] are clipped."""
    import soundfile as sf

    x = _check_signal(x)
    _check_fs(fs)
    try:
        sf.write(str(path), np.clip(x, -1.0, 1.0), int(fs), subtype="PCM_16")
    except Exception as e:
        raise ValueError(f"Cannot save audio to '{path}': {e}") from e


# ----------------------------------------------------------------------------
# Spectrum
# ----------------------------------------------------------------------------
def compute_fft(x, fs):
    """One-sided spectrum of x. Returns (freqs, magnitude, phase).

    freqs in Hz (0 .. fs/2), magnitude = |X|, phase in radians.
    """
    x = _check_signal(x)
    _check_fs(fs)
    X = fft(x)
    freqs = freq_axis(len(x), fs)
    freqs, X = one_sided(freqs, X)
    return freqs, magnitude(X), phase(X)


# ----------------------------------------------------------------------------
# STFT / spectrogram
# ----------------------------------------------------------------------------
def _frame_count(n, window_size, hop):
    return 1 + max(0, (n - window_size)) // hop


def _stft(x, window_size, hop):
    """Hand-written STFT. Returns complex array of shape (n_freqs, n_frames).

    Frame m covers samples [m*hop, m*hop + window_size). A short signal is
    zero-padded to one full window.
    """
    if window_size < 2:
        raise ValueError("window_size must be at least 2")
    if hop < 1:
        raise ValueError("hop must be at least 1")
    if len(x) < window_size:
        x = np.pad(x, (0, window_size - len(x)))
    w = _hamming(window_size)
    n_frames = _frame_count(len(x), window_size, hop)
    n_bins = window_size // 2 + 1
    S = np.empty((n_bins, n_frames), dtype=complex)
    for m in range(n_frames):
        frame = x[m * hop: m * hop + window_size] * w
        S[:, m] = fft(frame)[:n_bins]
    return S


def compute_spectrogram(x, fs, window_size=1024, hop=512):
    """Spectrogram by a manual STFT (Hamming window).

    Returns (times, freqs, S_db):
      times  : centre time of every frame, seconds
      freqs  : 0 .. fs/2 in Hz
      S_db   : 20*log10(|STFT|), shape (len(freqs), len(times))
    Plot with  pcolormesh(times, freqs, S_db).
    """
    x = _check_signal(x)
    _check_fs(fs)
    window_size, hop = int(window_size), int(hop)
    S = _stft(x, window_size, hop)
    n_frames = S.shape[1]
    times = (np.arange(n_frames) * hop + window_size / 2) / fs
    freqs = np.arange(window_size // 2 + 1) * fs / window_size
    return times, freqs, to_db(magnitude(S))


# ----------------------------------------------------------------------------
# Filters (LPF / HPF / BPF / Notch) on audio
# ----------------------------------------------------------------------------
def filter_audio(x, fs, filter_type, cutoff, order=None, width=2.0):
    """Frequency-domain filter on audio. Thin wrapper over core.filters.freq_filter_1d.

    filter_type: 'lowpass' | 'highpass' | 'bandpass' | 'notch'
    cutoff     : Hz (a (low, high) tuple for bandpass)
    order      : None = ideal brick-wall (can "ring"), int = smooth Butterworth
    Returns the filtered signal (same length as x, clipped to [-1, 1]).
    Cutoffs at or above fs/2 are rejected because they would do nothing.
    """
    x = _check_signal(x)
    _check_fs(fs)
    nyq = fs / 2.0
    cuts = cutoff if filter_type == "bandpass" and np.ndim(cutoff) == 1 else [cutoff]
    for c in np.atleast_1d(cuts):
        if np.ndim(c) == 0 and c >= nyq:
            raise ValueError(f"Cutoff {c} Hz must be below the Nyquist frequency ({nyq:g} Hz)")
    y, _, _ = freq_filter_1d(x, fs, filter_type, cutoff, order, width)
    return np.clip(y, -1.0, 1.0)


# ----------------------------------------------------------------------------
# Echo (convolution with an impulse response)
# ----------------------------------------------------------------------------
def add_echo(x, fs, delay_s, decay, n_echoes=1):
    """Add an echo by convolving x with an impulse response.

        h[0] = 1 (the direct sound),  h[k*d] = decay**k  (k = 1..n_echoes),
        d = round(delay_s * fs) samples.

    The output is longer than x by n_echoes*d samples so the echo tail is not cut.
    If the result peaks above 1 it is scaled down to peak 1 (no clipping).
    """
    x = _check_signal(x)
    _check_fs(fs)
    if delay_s <= 0:
        raise ValueError("Echo delay must be positive")
    if not 0 <= decay < 1:
        raise ValueError("Echo decay must be in [0, 1)")
    if int(n_echoes) < 1:
        raise ValueError("n_echoes must be at least 1")
    d = int(round(delay_s * fs))
    if d < 1:
        raise ValueError("Echo delay is shorter than one sample")
    n_echoes = int(n_echoes)
    if d * n_echoes > 20 * fs:
        raise ValueError("Echo is too long (more than 20 seconds)")

    h = np.zeros(d * n_echoes + 1)
    for k in range(n_echoes + 1):
        h[k * d] = decay ** k

    if len(h) <= 64:                      # tiny kernel: direct convolution is fine
        y = convolve1d(x, h, mode="full")
    else:                                 # long kernel: via FFT (same result, much faster)
        y = _fft_convolve(x, h)
    peak = np.max(np.abs(y))
    return y / peak if peak > 1.0 else y


# ----------------------------------------------------------------------------
# Noise reduction
# ----------------------------------------------------------------------------
def reduce_noise(x, fs, method="spectral_subtraction", **params):
    """Reduce stationary background noise.

    method = 'spectral_subtraction'
      For every STFT frame:  |Y| = max(|X| - alpha*|N|, beta*|X|)  and the
      ORIGINAL phase is kept; frames are put back with weighted overlap-add.
      |N| is the average noise magnitude, taken from (first match wins):
        noise_clip     : array of samples that contain only noise, or
        noise_duration : seconds at the START of x that contain only noise, or
        (default)      : the 10 % quietest frames of x (automatic). This needs some
                         quiet moments (pauses) in the recording. A signal that is
                         equally loud everywhere has no noise-only part, so the
                         method would treat the signal itself as noise.
      params (all optional):
        alpha=2.0   over-subtraction (bigger = removes more noise, more artefacts)
        beta=0.05   spectral floor (keeps a little signal so there is no "musical noise")
        window_size=1024, hop=256
    Returns an array of the same length as x.
    """
    x = _check_signal(x)
    _check_fs(fs)
    if method != "spectral_subtraction":
        raise ValueError("method must be 'spectral_subtraction'")

    alpha = float(params.get("alpha", 2.0))
    beta = float(params.get("beta", 0.05))
    window_size = int(params.get("window_size", 1024))
    hop = int(params.get("hop", 256))
    noise_clip = params.get("noise_clip")
    noise_duration = params.get("noise_duration")
    if alpha < 0:
        raise ValueError("alpha must be >= 0")
    if not 0 <= beta <= 1:
        raise ValueError("beta must be in [0, 1]")
    if window_size < 2 or hop < 1 or hop > window_size:
        raise ValueError("Need window_size >= 2 and 1 <= hop <= window_size")

    n = len(x)
    if n < window_size:
        raise ValueError("Signal is shorter than one analysis window; cannot reduce noise")
    w = _hamming(window_size)
    # pad so that every sample is covered by several frames
    pad_left = window_size - hop
    n_frames = _frame_count(n + pad_left, window_size, hop) + 1
    total = (n_frames - 1) * hop + window_size
    xp = np.zeros(total)
    xp[pad_left:pad_left + n] = x

    # --- analysis -----------------------------------------------------------
    spectra = np.empty((n_frames, window_size), dtype=complex)
    for m in range(n_frames):
        spectra[m] = fft(xp[m * hop: m * hop + window_size] * w)
    mags = magnitude(spectra)

    # Frames made only of REAL samples (no zero padding). Frames that touch the
    # padding look artificially quiet and must not be used to estimate noise.
    starts = np.arange(n_frames) * hop
    real = np.where((starts >= pad_left) & (starts + window_size <= pad_left + n))[0]

    # --- noise estimate -----------------------------------------------------
    if noise_clip is not None:
        nc = _check_signal(noise_clip, "noise_clip")
        if len(nc) < window_size:
            raise ValueError("noise_clip is shorter than one window")
        noise_mag = np.mean(
            [magnitude(fft(nc[i:i + window_size] * w))
             for i in range(0, len(nc) - window_size + 1, hop)], axis=0)
    elif noise_duration is not None:
        n_noise = int(noise_duration * fs)
        if noise_duration <= 0 or n_noise < window_size or n_noise > n:
            raise ValueError("noise_duration must be positive, at least one window long "
                             "and not longer than the signal")
        inside = real[starts[real] + window_size <= pad_left + n_noise]
        if len(inside) == 0:
            raise ValueError("noise_duration is too short to contain a full analysis frame")
        noise_mag = mags[inside].mean(axis=0)
    else:
        energy = np.sum(mags[real] ** 2, axis=1)
        k = max(1, int(0.1 * len(real)))
        noise_mag = mags[real[np.argsort(energy)[:k]]].mean(axis=0)

    # --- subtraction (original phase is kept) -------------------------------
    clean = np.maximum(mags - alpha * noise_mag, beta * mags)
    spectra = clean * np.exp(1j * phase(spectra))

    # --- weighted overlap-add synthesis -------------------------------------
    out = np.zeros(total)
    wsum = np.zeros(total)
    for m in range(n_frames):
        sl = slice(m * hop, m * hop + window_size)
        out[sl] += ifft(spectra[m]).real
        wsum[sl] += w
    out = out / np.maximum(wsum, 1e-3)
    return out[pad_left:pad_left + n]
