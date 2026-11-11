# CONTRACT.md: Shared Function Signatures

This file is the agreement between the Core and GUI members. **Do not change a signature without telling everyone** and updating this file in the same PR.

## General rules

- All signals are `numpy.ndarray` of dtype `float64` (images: values in `[0, 255]`, 2D array, grayscale).
- Complex spectra are `numpy.ndarray` of dtype `complex128`.
- Every function has a docstring and returns exactly what is written here.
- Angles (phase) are in **radians**, in `[-pi, pi]`.
- Frequencies are in **Hz**, sampling rate `fs` in **Hz**.
- Functions never open GUI windows or print plots; they only return data.
- Invalid input raises `ValueError` with a clear message (the GUI catches it and shows a message box).

---

## `core/fourier.py`  (owner: Moaz)

```python
def dft(x: np.ndarray) -> np.ndarray:
    """O(N^2) DFT. Returns complex array, same length as x."""

def idft(X: np.ndarray) -> np.ndarray:
    """Inverse DFT. Returns complex array (take .real for real signals)."""

def fft(x: np.ndarray, n: int | None = None) -> np.ndarray:
    """Cooley-Tukey FFT. Output length == len(x) (any length works), or n if given."""

def ifft(X: np.ndarray, n: int | None = None) -> np.ndarray:
    """Inverse FFT. Returns complex; take .real for real signals."""

def fft2(img: np.ndarray) -> np.ndarray:
    """2D FFT (rows then columns). NOT shifted."""

def ifft2(F: np.ndarray) -> np.ndarray:
    """2D inverse FFT. Returns complex array."""
```

## `core/convolution.py`  (owner: Mohab)

```python
def convolve1d(x: np.ndarray, h: np.ndarray, mode: str = "same") -> np.ndarray:
    """mode: 'full' | 'same' | 'valid'. Same behaviour as numpy.convolve."""

def convolve2d(img: np.ndarray, kernel: np.ndarray, padding: str = "zero") -> np.ndarray:
    """Output has the same shape as img. padding: 'zero' | 'reflect' | 'edge'."""
```

## `core/utils.py`  (owner: Omar)

```python
def magnitude(X: np.ndarray) -> np.ndarray:        # |X|
def phase(X: np.ndarray) -> np.ndarray:            # angle(X), radians
def to_db(m: np.ndarray, eps: float = 1e-12) -> np.ndarray:   # 20*log10(m + eps)
def normalize(x: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
def freq_axis(n: int, fs: float, shift: bool = False) -> np.ndarray:
    """Frequency values in Hz for an n-point FFT."""
def fftshift(X, axes=None) -> np.ndarray      # 1D or 2D, same as numpy.fft.fftshift
def ifftshift(X, axes=None) -> np.ndarray
def one_sided(freqs, *spectra)                # keep bins 0..N/2 only
```

## `core/filters.py`  (owner: Fatma)

```python
def freq_filter_1d(x, fs, filter_type, cutoff, order=None, width=2.0):
    """
    filter_type: 'lowpass' | 'highpass' | 'bandpass' | 'notch'
    cutoff: float for lowpass/highpass/notch, (low, high) tuple for bandpass.
    order: None = ideal brick-wall, int = Butterworth.  width: notch width in Hz.
    Returns: (y, X, Y)  -> filtered signal, input spectrum, output spectrum
    """

def freq_mask_2d(shape, filter_type, cutoff, kind="ideal", order=2):
    """
    shape: (rows, cols). filter_type: 'lowpass' | 'highpass'.
    kind: 'ideal' | 'gaussian' | 'butterworth'.
    Returns a real mask in [0, 1], centered (to be used with fftshift'ed spectrum).
    """

def freq_filter_2d(img, filter_type, cutoff, kind="ideal", order=2):
    """Returns (out_img, F_shifted, G_shifted): output, input spectrum, output spectrum."""
```

## Unified filter entry point: `core/filters.py`

```python
def apply_filter(signal, filter_type: str, **params) -> tuple:
    """
    Returns: (output, magnitude, phase)
      output    : filtered signal (same shape as input)
      magnitude : magnitude spectrum of the OUTPUT
      phase     : phase spectrum of the OUTPUT
    1D: params = fs, cutoff, [order], [width]   (magnitude/phase in raw FFT order, length N)
    2D: params = cutoff, [kind], [order]        (magnitude = log1p, centered; phase centered)
    """
```

---

## `created_signal/processing.py`  (owner: Moaz, used by Omar)

```python
def generate_signal(kind: str, amp: float, freq: float, phase: float,
                    fs: float, duration: float) -> tuple[np.ndarray, np.ndarray]:
    """kind: 'sine' | 'cosine' | 'square' | 'triangle' | 'sawtooth'. Returns (t, x)."""

def combine_signals(signals: list[np.ndarray]) -> np.ndarray
def add_noise(x: np.ndarray, snr_db: float) -> np.ndarray
def apply_conv_filter(x: np.ndarray, kernel_type: str, size: int = 5,
                      kernel: np.ndarray | None = None) -> np.ndarray:
    """kernel_type: 'moving_average' | 'difference' | 'custom'."""

def analyze(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (freqs, magnitude, phase), one-sided spectrum."""
```

## `image_signal/processing.py`  (owner: Mohab, used by Hagar)

```python
def load_image(path: str) -> np.ndarray:            # grayscale float64 [0, 255]
def spatial_filter(img, name: str, size: int = 3, sigma: float = 1.0) -> np.ndarray:
    """name: 'box' | 'gaussian' | 'sharpen' | 'sobel_x' | 'sobel_y' | 'sobel'
             | 'prewitt' | 'laplacian'"""
def image_fft(img) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (F_shifted, log_magnitude, phase)."""
def frequency_filter(img, filter_type, cutoff, kind="ideal", order=2) -> tuple:
    """Returns (output_img, log_magnitude_of_output, phase_of_output)."""
```

## `speech_signal/processing.py`  (owner: Fatma, used by Ghada)

```python
def load_audio(path: str) -> tuple[np.ndarray, int]:    # (mono signal in [-1, 1], fs)
def record_audio(duration: float, fs: int = 44100) -> np.ndarray
def save_audio(path: str, x: np.ndarray, fs: int) -> None
def compute_fft(x, fs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (freqs, magnitude, phase), one-sided."""
def compute_spectrogram(x, fs, window_size=1024, hop=512) -> tuple:
    """Returns (times, freqs, S_db). Hamming window, manual STFT."""
def add_echo(x, fs, delay_s: float, decay: float) -> np.ndarray   # via convolution
def reduce_noise(x, fs, method="spectral_subtraction", **params) -> np.ndarray
```

---

## Testing rule

Every function in `core/` must have a test in `tests/` that compares it to `numpy` / `scipy` with `np.allclose(..., atol=1e-6)`.

## Change process

1. Open an Issue: "Contract change: <function>".
2. Get a thumbs-up from the owner and the GUI member who uses it.
3. Update this file and the code in the same Pull Request.
