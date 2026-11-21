# DSP Project Report: Multi-Domain Signal Processing

**Team (6 students, three pairs):**

| Domain | Core (algorithms) | GUI | 
|---|---|---|
| Created Signal |  Omar (Mathematical Signal & DTFT), Moez(Convolution & Difference Equations) |Omar
| Speech / Audio |  Ghada (Audio Analysis), Fatma(Audio Filtering) |Ghada
| Image |           Hager (2D Fourier & Image Filtering), Mohab (Image Noise Removal & Evaluation) |Mohab

---

## 1. Summary

We built one Python desktop application (PyQt5 + matplotlib) with three tabs, one per domain. Each tab lets the user create or load a signal, filter it, and look at the magnitude, the phase and the result before and after processing. All the heavy algorithms (FFT, 2D FFT, convolution, frequency-domain filters, DTFT, IIR notch, spectral subtraction) are written by us in `core/` and the module folders, and every one of them is checked against NumPy / SciPy.

| Measure | Result |
|---|---|
| Programming tasks of the brief that pass an automatic check | **19 / 19** (`python scripts/check_brief.py`) |
| Automatic tests | **514** (513 pass, 1 skipped on purpose), about 30 s to run |
| Code | about 4,400 lines of source and 2,800 lines of tests |
| Worst difference from NumPy / SciPy | about `7e-13` (Created), about `2e-10` (Image), `0` to `1e-12` (Speech filters) |
| Restoration of the periodic-noise image (synthetic test, 256x256) | MSE 450 → 0.80, PSNR 21.6 → 49.1 dB |

---

## 2. Compliance with the brief

### Omar: Mathematical Signal & DTFT

| Requirement | Where | Evidence |
|---|---|---|
| Two sinusoids + Gaussian noise | Created tab: Components 1 and 2 + Gaussian noise box | peaks at 5 and 40 Hz, measured SNR 15.1 dB (set 15) |
| DTFT from the defining equation, with loops | `created_signal/processing.py: dtft` (double loop, no FFT inside), tab "DTFT (loops)" | error against the FFT `2.5e-13` |
| Magnitude and phase spectra | "Signals" tab (Hz) and "DTFT (loops)" tab (rad/sample) | 256 points in [-π, π] |

### Moez: Convolution & Difference Equations

| Requirement | Where | Evidence |
|---|---|---|
| Moving average by linear convolution | filter "Moving average (convolution)" | `4e-16` from `np.convolve` |
| Time-domain convolution vs frequency-domain multiplication, zero-padding | tab "Convolution vs FFT" | zero-padded error `1.6e-15`; without padding (circular) error 0.46 |
| First-order difference equation | filter `y[n] = a·y[n-1] + b·x[n]` | identical to `scipy.signal.lfilter` |

### Ghada: Audio Analysis

| Requirement | Where | Evidence |
|---|---|---|
| Load a WAV file | `speech_signal/processing.py: load_audio`, "Load audio file" | 24,000 samples at 8,000 Hz |
| Add a 50 or 60 Hz interference tone | `add_tone`, buttons "50 Hz" / "60 Hz" | measured amplitude 0.200 (set 0.2) |
| Clean and noisy waveforms and spectra; identify the interference frequency | "Compare with: Original", `find_peak_frequency` | identified 50.00 Hz |

### Fatma: Audio Filtering

| Requirement | Where | Evidence |
|---|---|---|
| Second-order notch, coefficients calculated for the chosen frequency | `notch_coefficients`; the GUI prints `b` and `a` | 50 Hz: `b = [1, -1.9985, 1]`, `a = [1, -1.9906, 0.9922]`; zeros on the unit circle, poles inside (stable) |
| Process the audio, reduce the tone | `apply_notch` (difference equation, sample by sample) | 50 Hz reduced by 162 dB in steady state, speech tone kept |
| Save the filtered WAV, compare spectra and listening | `save_audio`, overlay plots, spectrogram, Play Original / Processed | WAV reloads with the same samples (16-bit rounding only) |

### Hager: 2D Fourier & Image Filtering

| Requirement | Where | Evidence |
|---|---|---|
| Load an image and convert it to grayscale | `image_signal/processing.py: load_image` | RGB, RGBA, palette, 1-bit, 16-bit and float files all give a 2D 0–255 array |
| 2D spectrum with fft2 and fftshift | `image_fft`, "Analyze FFT" | error against NumPy `4.7e-10`, DC at the centre |
| Ideal circular low-pass and high-pass | `frequency_filter(kind="ideal")` | mask is a disc, not a square; low-pass blurs, high-pass keeps edges |

### Mohab: Image Noise Removal & Evaluation

| Requirement | Where | Evidence |
|---|---|---|
| Periodic sinusoidal noise | `denoising.py: add_periodic_noise` | noise = `30·sin(2π(8x/W + 6y/H))` exactly |
| Noisy spectrum and a Notch Mask from the known frequency | "Add Periodic Noise", `create_notch_mask` | noise peaks are 813 times stronger than in the clean image; the mask removes 58 pixels at the two peaks and keeps DC |
| Inverse 2D FFT; compare original, noisy, restored | "Remove Periodic Noise": original / restored / noisy side by side, both spectra and the mask | RMSE 21.2 → 0.6 |
| MSE and PSNR | `image_signal/metrics.py`; status line | MSE 450 → 0.34, PSNR 21.6 → 52.8 dB (128x128 test) |

The brief also says the manual implementation is only *required* for the DTFT; we wrote the other algorithms by hand as well and compared them with the libraries.

---

## 3. Architecture

```
dsp_project/
├── main.py                  # window with 3 tabs; each tab class is loaded from its own package
├── core/                    # shared algorithms, no GUI code
│   ├── fourier.py           # DFT, FFT, IFFT, FFT2, IFFT2
│   ├── convolution.py       # 1D and 2D convolution
│   ├── filters.py           # frequency-domain filters (1D and 2D masks)
│   ├── utils.py             # magnitude, phase, dB, fftshift, frequency axis
│   └── gui_widgets.py       # PlotCanvas, ImageCanvas, LabeledSlider
├── created_signal/          # Students 1, 2: processing, reference (numpy/scipy), compare, gui
├── speech_signal/           # Students 3, 4: processing, gui
├── image_signal/            # Students 5, 6: processing, denoising, metrics, gui
├── scripts/                 # check_brief.py, compare_image.py
├── tests/                   # 514 tests
└── docs/screenshots/        # figures used in this report
```

**Design rules we kept:**

1. **No signal processing inside a GUI file.** A GUI only reads the controls, calls one function, catches `ValueError` and draws the returned arrays.
2. **Every processing function is pure** (arrays in, arrays out, no windows or plots), so it can be tested without a screen.
3. **Errors are `ValueError` with a readable message.** The GUI shows it in a message box or in the status line; nothing crashes on bad input.
4. **One contract** (`CONTRACT.md`) lists every shared function signature.

### 3.1 The FFT we wrote

- **Radix-2 Cooley–Tukey**, decimation in time: split into even and odd samples while the length is even.
- Pieces of length 64 or less are finished with a small cached DFT matrix.
- A long **odd** piece is handled by **Bluestein's chirp-z algorithm**, which turns the DFT into a convolution computed with power-of-two FFTs. So *any* length works, with no padding that would change the signal.
- The inverse FFT uses `ifft(X) = conj(fft(conj(X))) / N`; the 2D FFT is a 1D FFT on every row, then on every column.

---

## 4. Module 1: Created Signal (Moez and Omar)

**What it does**

- Five waveforms (sine, cosine, square, triangle, sawtooth) with amplitude, frequency, phase, sampling rate and duration; up to three components added together; Gaussian noise set by SNR in dB.
- Filters: moving average, difference, Gaussian, custom kernel (all by convolution); first-order difference equation; low-pass, high-pass, band-pass and notch in the frequency domain (ideal or Butterworth).
- Four tabs: **Signals** (time, magnitude, phase), **Accuracy** (ours vs NumPy/SciPy), **DTFT (loops)**, **Convolution vs FFT**.

**Accuracy against NumPy / SciPy** (default setup: 5 Hz + 40 Hz + 15 dB noise, Fs 1000 Hz, 1 s)

| Operation | Reference | Max error |
|---|---|---|
| Waveform generation | numpy / scipy.signal | `0` |
| FFT magnitude | `numpy.fft.rfft` | `1.4e-16` |
| FFT phase | `numpy.angle` | `1.2e-13` |
| IFFT(FFT(x)) | `numpy.fft.ifft` | `1.9e-15` |
| DTFT (loops, 128 samples) | `numpy.fft.fft` | `6.8e-13` |
| Convolution (moving average) | `scipy.signal.convolve` | `2.2e-16` |
| Linear convolution via FFT (zero-padded) | `numpy.convolve` | `1.9e-15` |
| Difference equation | `scipy.signal.lfilter` | `0` |
| Fourier filter (low-pass) | `numpy.fft` | `1.2e-15` |

Our code is slower than NumPy for the FFT (about 20 times on 1000 samples) and about 400 times slower for the DTFT written with loops, because NumPy is compiled C. That is expected and fine for a teaching project. Timings come from one run on one machine.


The orange dots on the DTFT plot are the DFT computed by the FFT: they sit exactly on the curve, because the DFT is the DTFT sampled at `ω = 2πk/N`.


Without zero-padding the product of FFTs is a **circular** convolution: the tail wraps around and corrupts the first `M-1` samples (the spike on the lower plot).

---

## 5. Module 2: Speech (Ghada and Fatma)

**What it does**

- Load a WAV/FLAC/OGG file (stereo is averaged to mono) or record from the microphone in a separate thread; Play Original / Processed / Stop; Save the processed audio as a 16-bit WAV.
- Add a 50 or 60 Hz tone, find the strongest frequency in a search range (a red marker is drawn on the spectrum).
- Filters: low-pass, high-pass, band-pass ("Telephone band" preset), frequency notch, and the **second-order IIR notch**.
- Echo (convolution with an impulse response), noise reduction (spectral subtraction), spectrogram (manual STFT, Hamming window).

**The notch filter**

```
H(z) = (1 - 2cos(w0) z^-1 + z^-2) / (1 - 2 r cos(w0) z^-1 + r^2 z^-2),     w0 = 2*pi*f0/fs,   r = 1 - pi*BW/fs
y[n] = b0 x[n] + b1 x[n-1] + b2 x[n-2] - a1 y[n-1] - a2 y[n-2]
```

The zeros sit on the unit circle at `e^{±jw0}`, so the gain at `f0` is exactly zero. The poles sit at radius `r` just inside the circle, so only a narrow band around `f0` is affected, and the filter is stable because `|r| < 1`.

**Measured behaviour** (speech-like test signal with a 0.2 amplitude 50 Hz hum, Fs 8 kHz, 3 s)

| Notch bandwidth | Reduction of the 50 Hz line over the whole file | After the first second | Start-up ringing above 0.01 lasts |
|---|---|---|---|
| 5 Hz | 33.5 dB | more than 160 dB | 186 ms |
| 10 Hz | 39.5 dB | more than 160 dB | 95 ms |
| 20 Hz | 45.5 dB | more than 160 dB | 46 ms |

The filter starts from zero state, so the hum switching on at `n = 0` leaves a short ringing at the start of the file. A narrower notch removes more of the neighbouring spectrum less but rings longer. This is a trade-off worth explaining in the discussion, and it is why the spectrum of the whole file still shows a small bump at 50 Hz.


**Speed on 10 s of 44.1 kHz audio:** FFT 0.6 s, spectrogram 0.1 s, IIR notch 0.16 s, frequency filter 0.9 s, echo 0.3 s, noise reduction 0.4 s.


---

## 6. Module 3: Image (Hager and Mohab)

**What it does**

- Load any common image file and convert it to grayscale.
- Spatial filters: box, Gaussian, sharpen, Sobel X / Y / magnitude, Prewitt, Laplacian (kernel size 3 to 9).
- 2D FFT with `fftshift`: log magnitude and phase. Frequency filters: ideal, Gaussian, Butterworth, low-pass and high-pass.
- Periodic noise, noisy spectrum, notch mask, restoration, MSE / RMSE / PSNR.

**Noise removal**

After "Remove Periodic Noise" the window shows everything the brief asks to compare: original, restored and noisy images in the top row; the restored spectrum (black dots where the notch removed the noise peaks), the noisy spectrum and the notch mask in the bottom row. The status line reports the result before and after:

`Restored | MSE: 0.80 | RMSE: 0.89 | PSNR: 49.10 dB  (noisy: MSE 450.00, PSNR 21.60 dB)`

| Test | MSE noisy → restored | PSNR noisy → restored |
|---|---|---|
| 256x256 synthetic picture, amplitude 30, fx = 8, fy = 6, radius 3 | 450 → 0.80 | 21.6 → 49.1 dB |
| 128x128 synthetic picture, same noise | 450 → 0.34 | 21.6 → 52.8 dB |

The MSE of the noisy image is exactly `A²/2 = 450`, which is the mean square of a sine of amplitude 30. These numbers come from synthetic pictures; a real photo has more energy at the two removed bins, so its PSNR gain will be smaller. The brief explicitly does not ask for a fixed dB target; we report what we achieve.

**Comparison with NumPy / SciPy** (`scripts/compare_image.py`, 29 sections: 8 spatial filters, the FFT magnitude / phase / complex spectrum, and 6 frequency filters with their magnitude and phase): the largest difference is about `2e-10` pixel levels. Spatial filters are compared away from the border, because our reflect padding (`abcd|cba`) and SciPy's `symm` padding (`abcd|dcba`) are different conventions.

**Speed:** 512x512 FFT 0.03 s; 1920x1080 FFT 0.9 s, frequency filter 1.8 s, Gaussian 9x9 0.3 s.

---

## 7. Testing

**514 automatic tests** (513 pass; one is skipped on purpose because NumPy swaps its arguments in that case). All GUI tests run without a screen, a microphone or a sound card.

| Test file | Tests | What it covers |
|---|---|---|
| `test_fourier.py` | 50 | DFT, FFT, IFFT, FFT2 against NumPy for even, odd, prime and audio-size lengths |
| `test_convolution.py` | 30 | 1D (full, same, valid) and 2D convolution against NumPy / SciPy, padding modes |
| `test_filters.py`, `test_utils.py`, `test_gui.py` | 28 | frequency filters (1D and 2D masks), helpers, shared widgets and the main window |
| `test_created_processing.py` | 26 | waveforms, noise SNR, analysis, filters, aliasing, pipeline |
| `test_created_project_spec.py` | 14 | DTFT with loops, convolution vs FFT, difference equation |
| `test_created_compare.py` | 36 | numpy/scipy comparison engine, including a test that it fails when the code is broken |
| `test_created_gui.py` | 25 | all four tabs, warnings, errors, export, phase plot |
| `test_speech_processing.py` | 84 | audio I/O, spectrum, spectrogram, filters, echo, noise reduction, tone, peak, notch coefficients and response |
| `test_speech_gui.py` | 48 | the whole speech tab, recording with a fake device, plot fixes |
| `test_image_processing.py` | 80 | loading every image mode, 8 spatial filters, FFT, frequency filters |
| `test_image_denoising.py` | 44 | noise, peaks, notch mask, restoration, metrics, frequency limits |
| `test_image_gui.py` | 48 | the whole image tab, three-column comparison, save |
| `test_brief_check.py` | 1 | all 19 programming tasks of the brief |


---

## 8. Problems we met and how they were solved

| Problem | Cause | Fix |
|---|---|---|
| FFT very slow on 3 s of audio (about 8 s) | the length splits down to a large odd piece, which used a full DFT | Bluestein for large odd lengths, cached small DFT matrix: about 0.3 s |
| A bin exactly on the cutoff could be dropped | rounding in `f <= fc` | relative tolerance `1e-9` in the filter and the reference |
| Phase differs by `2π` from NumPy | `-π` and `+π` are the same angle | circular error in the comparison |
| Phase plot looked empty | weak bins hidden left isolated points, and matplotlib draws no line through one point | draw phase as dots, always keep the strongest bin, threshold slider |
| Magnitude plot flattened when zoomed (speech) | the y-axis kept the range of the whole spectrum | y-axis fitted to the visible range |
| 16-bit grayscale images loaded all white | PIL clips values above 255 | rescale 16-bit, 32-bit and float images |
| Periodic noise above half the image size aliased silently, so nothing was restored | the notch was built where the noise no longer was | frequency validation in the core, and the spin boxes limit themselves to the image size |
| Loading a new image kept the old noisy one | state was not reset | `noisy_image` cleared on load |
| Noisy image was not visible after restoration | only four canvases | third column: noisy image and notch mask; both spectra |
| MSE was missing | status line showed RMSE only | `image_metrics`, MSE and PSNR before and after |
| Comparison script showed an error of about 265 for Sobel X / Y | reference clipped to 0–255, ours is signed | reference no longer clips signed Sobel |
| Signed Sobel results lost their negative half when saved | PNG holds 0–255 | rescale signed results; round instead of truncate |
| The GUI depended on a `scripts/` file for its metrics | a helper lived in a script | moved to `image_signal/metrics.py` |

---



## 9. Limitations

- Created Signal: three components at once in the GUI; Sampling rate up to 10 kHz and duration up to 5 s so the live update stays smooth. Noise is off by default (tick the box for the brief's exact signal).
- Speech: the IIR notch rings briefly at the start of a file (Section 5). Spectral-subtraction noise reduction needs quiet moments in the recording, or a known noise-only segment.
- Image: the frequency of the periodic noise must be known and below half the image size. High-pass results are clipped to 0–255, so they look mostly dark.
- Our FFT is slower than NumPy's (Section 4). Everything is single-threaded and runs in the GUI thread except recording.
- Numbers in this report come from one machine and synthetic test data; re-run `python scripts/check_brief.py` and the tests on your own computer before the demo.

---

## 10. How to run

```bash
pip install -r requirements.txt
python main.py                      # the application
python -m pytest -q                 # 514 tests (about 30 s)
python scripts/check_brief.py       # the 19 programming tasks of the brief
python scripts/compare_image.py     # image filters vs numpy/scipy (put a test_image.jpg in scripts/)
```

## 11. Review notes: what changed during integration

So nobody is surprised when comparing with their own copy:

- **Speech GUI** (`speech_signal/gui.py`): the magnitude y-axis now fits the zoomed range; the phase is drawn as dots on the strong bins with a "Phase threshold" control. No other change to the team's code.
- **Image core and GUI:** 16-bit loading; frequency validation; `image_signal/metrics.py` added; the GUI shows the third column, MSE, and limits the noise frequencies to the image size; saving rescales signed results. `scripts/compare_image.py` imports its metrics from the package and no longer clips signed Sobel in the reference. `image_signal/__init__.py` and `scripts/__init__.py` were written during integration.
- **Added by the review:** the test suite for Speech and Image, `scripts/check_brief.py`, `BRIEF_CHECKLIST.md`, and this report.

---

## 12. Conclusion

The three modules together cover all six roles of the brief: a DTFT written with loops, convolution and difference equations, WAV analysis with a calculated second-order notch, 2D Fourier filtering, and periodic-noise removal evaluated with MSE and PSNR. The hand-written code agrees with NumPy and SciPy to between `1e-16` and `1e-10` in every comparison we made, the 19 programming tasks pass an automatic check, and the application runs as one program. What remains is to run it on a real computer with a microphone, speakers, the instructor's files, and for every student to be ready to explain their own part.
