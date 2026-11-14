# Created Signal Module Report: DSP Project

**Owners:** Moaz (Core: generation, filters, analysis, comparison) and Omar (GUI)
**Brief roles covered:**  (Mathematical Signal & DTFT) +  (Convolution & Difference Equations)

---

## 1. Objective

Build a Python application that generates a 1D digital signal from user-defined parameters, applies filters from the DSP course (convolution, Fourier-based filters, and a difference equation), and computes and displays the magnitude, the phase, and the DTFT. It also proves the accuracy of our hand-written code by comparing it against numpy/scipy inside the GUI itself.

### Requirements traceability (project brief)

| Requirement | Where in the project | Status |
|---|---|---|
| Signal with two sinusoids + Gaussian noise | Two components (5 Hz and 40 Hz) + noise box | Done |
| DTFT from its defining equation, using loops, for a limited number of samples and frequency points | `processing.dtft` + "DTFT (loops)" tab | Done |
| Magnitude and phase plots | "Signals" tab (in Hz) and "DTFT" tab (in rad/sample) | Done |
| Moving average using linear convolution | Moving average filter (`convolve1d`) | Done |
| Compare time-domain convolution with frequency-domain multiplication, with zero-padding | "Convolution vs FFT" tab | Done |
| Simple first-order difference equation | Filter `y[n] = a·y[n-1] + b·x[n]` | Done |
| Compare hand-written code with numpy/scipy (our addition) | "Accuracy" tab (9 operations) | Done |

---

## 2. Code structure

| File | Purpose |
|---|---|
| `core/fourier.py` | Hand-written DFT, FFT, IFFT, FFT2 (shared with the other teams) |
| `core/convolution.py` | Hand-written 1D and 2D convolution |
| `core/filters.py` | Frequency-domain filters (ideal and Butterworth) |
| `core/utils.py` | magnitude, phase, to_db, freq_axis, fftshift |
| `core/gui_widgets.py` | `PlotCanvas`, `LabeledSlider` and other shared widgets |
| `created_signal/processing.py` | Generation, noise, filters, analysis, DTFT, difference equation, pipeline |
| `created_signal/reference.py` | The same operations written with numpy/scipy (independent code) |
| `created_signal/compare.py` | Comparison engine (errors and timing) |
| `created_signal/gui.py` | `CreatedSignalTab` UI with four tabs |
| `tests/` | Tests comparing everything against numpy/scipy |

**Design principle:** all computation lives in pure functions with no GUI code, and the GUI only handles display. This made every part easy to test in isolation, and no mocks were ever needed because the GUI was wired to the real functions from the start.

---

## 3. Moaz's part: the Core

### 3.1 Signal generation

Time is `t = n / fs` and the angle is `θ = 2π·f·t + φ` (radians inside the Core; the GUI converts from degrees).

| Type | Formula |
|---|---|
| Sine | `A·sin(θ)` |
| Cosine | `A·cos(θ)` |
| Square | `+A` when `θ mod 2π < π`, otherwise `-A` |
| Triangle | `A·(2/π)·arcsin(sin θ)` |
| Sawtooth | `A·((θ mod 2π)/π − 1)` |

The function validates its inputs (positive sampling rate, a duration that gives at least two samples, a non-negative frequency) and raises a `ValueError` with a clear message instead of crashing the program.

### 3.2 Combining signals and noise

- `combine_signals`: sums signals of the same length.
- `add_noise(x, snr_db, seed)`: Gaussian noise with power `P_noise = P_signal / 10^(SNR/10)`. The seed stays fixed so the result is repeatable while the user moves sliders, and the **New noise** button changes it. A test confirms the measured SNR is within 0.5 dB of the requested one.

### 3.3 Convolution filters (time domain)

Formula: `y[n] = Σ h[k]·x[n−k]`. It is implemented as shift-and-add (for every kernel tap we add a scaled, shifted copy of the signal). This is exactly the same mathematical sum, only vectorized for speed, and our code never calls `np.convolve` or scipy. The output has the same length as the input (`same`) and is centered the same way numpy does it.

| Kernel | Effect |
|---|---|
| Moving average | Smoothing (a low-pass without an explicit cutoff) |
| Gaussian | Smoother smoothing with no ripple in the spectrum |
| Difference `[1, -1]` | Derivative (emphasizes fast changes) |
| Custom | Values typed by the user |

### 3.4 First-order difference equation

```
y[n] = a·y[n-1] + b·x[n]        (initial condition y[-1] = 0)
```

- A **recursive (IIR)** system, written with an explicit loop.
- Impulse response: `h[n] = b·aⁿ` for `n ≥ 0`.
- **Stable only when `|a| < 1`.** The GUI shows an UNSTABLE warning if `|a| ≥ 1`.
- DC gain = `b / (1 − a)`. For example `a = 0.9` and `b = 0.1` give a gain of 1.
- Verified by tests: the impulse response, the step response settling at `b/(1−a)`, and the output blowing up when `a > 1`.

### 3.5 Fourier filters (frequency domain)

Steps: `FFT → multiply by a mask H(f) → IFFT`. The mask is computed from `|f|` (negative frequencies are treated like the positive ones so the output stays real).

| Filter | Mask |
|---|---|
| Low-pass | Ideal: `1` when `f ≤ fc`. Butterworth: `1/(1+(f/fc)^(2n))` |
| High-pass | `1 − Low-pass` |
| Band-pass | `1` between `f_low` and `f_high` |
| Notch | `0` around a given frequency with a given width |

Order 0 means an ideal (brick-wall) filter, and any other number means a Butterworth filter of that order.

### 3.6 Analysis: magnitude and phase

`analyze(x, fs)` computes our own FFT and returns the non-negative half of the spectrum:

- **Magnitude:** `|X[k]|/N`, multiplied by 2 for the non-zero frequencies, so a sine of amplitude A shows a peak of exactly A.
- **Phase:** `arctan2(Im, Re)` in radians within `[−π, π]`. Bins with negligible energy get phase 0 because their phase is just numerical noise.

We verified that the peak sits at the right frequency with the right amplitude, that a cosine with a given phase shows that same value in the phase spectrum, and that a 700 Hz tone sampled at 1000 Hz shows up at 300 Hz (aliasing).

### 3.7 DTFT from the definition (Moez)

```
X(ω) = Σ  x[n] · e^(−jωn)          n = 0 .. N−1
```

- Written as an explicit **double loop** (an outer loop over frequency points, an inner loop over samples) using `math.cos` and `math.sin`, with no FFT anywhere.
- ω is in rad/sample from `−π` to `+π`. The link to real frequency is `ω = 2πf/fs`, and `ω = π` is the Nyquist frequency.
- To protect the UI, the function refuses to run when the number of operations (samples × points) exceeds 1.5 million, instead of freezing.
- **Verification:** the DTFT at `ω_k = 2πk/N` equals the DFT exactly. It was compared with `numpy.fft.fft` (error about `7e-13`). Tests also confirm that the DTFT of an impulse is flat, that a delayed impulse has linear phase, that a cosine peaks at the right ω with height `N/2`, and that a real signal has even magnitude and odd phase.

### 3.8 Convolution vs frequency-domain multiplication (Omar)

`convolution_vs_frequency(x, h)` returns:

| Item | Meaning |
|---|---|
| `direct` | Linear convolution in time (length `N+M−1`) |
| `padded` | `IFFT( FFT(x, L) · FFT(h, L) )` with `L = N+M−1` (zero-padding) |
| `circular` | The same without zero-padding (length `max(N, M)`) |

**Proven result:**

- With zero-padding it matches linear convolution (error about `1e-15`).
- Without zero-padding it becomes a **circular convolution**: the tail of the result wraps around and adds onto the first `M−1` samples, so the error is concentrated there (about 0.3 in our example) and every other sample is correct. This is exactly the role of zero-padding that the brief asks us to explain.

---

## 4. Omar's part: the GUI

The interface is built with PyQt5 and matplotlib: a scrollable control panel on the left and four tabs on the right.

### 4.1 Control panel

- **Sampling and display:** sampling rate (100 to 10,000 Hz), duration (up to 5 s), and the phase threshold.
- **3 Components:** each has an on/off switch, a waveform type, amplitude, frequency and phase (in degrees).
- **Gaussian noise:** enable, SNR, and a New noise button.
- **Filter:** a list of all filters (plus None): moving average, difference, Gaussian, custom, difference equation, low/high/band-pass and notch. Fields appear and disappear depending on the filter (for example `a` and `b` for the difference equation, and the cutoff for frequency filters).
- **Buttons:** Auto update, Apply, Reset, Save CSV, Save plots (PNG).
- Live updating uses a small delay (150 ms) so the plots do not lag while a slider is being dragged.

### 4.2 Signals tab (3 plots)

1. **Time domain:** the signal before the filter (Input) and after it (Output).
2. **Magnitude spectrum:** in Hz, with the range adjusted automatically around the highest frequency or cutoff.
3. **Phase spectrum:** dots on the bins stronger than a chosen fraction (default 5%) of the peak. The strongest bin is always shown, so the plot is never empty.

### 4.3 Accuracy tab

A table of **9 operations** with three error plots below it, showing `|ours − reference|` on a log scale. Details are in section 5.

### 4.4 DTFT (loops) tab

- Sliders for the number of samples used (N) and the number of frequency points.
- Two plots, magnitude and phase against ω in rad/sample, with the axis marked `−π … π`.
- Orange dots over the DTFT curve are the DFT computed by the FFT, showing that they are the same DTFT sampled at `ω_k`.
- An info line explains `ω = 2πf/fs` and where a 5 Hz tone sits on the axis.

### 4.5 Convolution vs FFT tab

- Top plot: the time-domain convolution with the zero-padded frequency-domain product on top of it (they coincide).
- Bottom plot: the error on a log scale for two versions: with zero-padding (about `1e-15`) and without it (a large spike in the first `M−1` samples).
- An info line gives the numbers and explains circular convolution.

### 4.6 Error handling

Any `ValueError` from the Core (all components disabled, letters in the kernel, a DTFT that is too large) is shown as a red message in the status area instead of crashing. If a component's frequency is above Nyquist, an aliasing warning appears.

---

## 5. Comparing our code with numpy/scipy

### 5.1 Method

We wrote a second version of every operation in `reference.py` using different libraries and without copying our code, so the comparison is meaningful. Then both versions run on **exactly the same signal** (noise included) and we measure:

- **Max |error|:** the largest difference at any point.
- **RMSE:** root mean square error.
- **Relative error:** the size of the error relative to the reference values.
- **Time:** the best of 3 runs (ms).
- **OK:** passes when the maximum error is below `1e-8` times the size of the values.

Phase is compared with a circular difference (`−π` and `+π` are the same angle), and only on bins that carry real energy.

### 5.2 Results (default setup: 5 Hz sine + 40 Hz sine + 15 dB noise, Fs = 1000 Hz, one second)

| Operation | Reference | Max error | Time (ours / reference) |
|---|---|---|---|
| Waveform generation | numpy / scipy.signal | `0` | 0.03 / 0.03 ms |
| FFT magnitude | `numpy.fft.rfft` | `1.4e-16` | 0.50 / 0.02 ms |
| FFT phase | `numpy.angle(rfft)` | `1.2e-13` | 0.50 / 0.02 ms |
| IFFT(FFT(x)) | `numpy.fft.ifft` | `1.9e-15` | 0.93 / 0.03 ms |
| DTFT (loops, 128 samples) | `numpy.fft.fft` | `6.8e-13` | 1.59 / 0.004 ms |
| Convolution (moving average) | `scipy.signal.convolve` | `2.2e-16` | 0.01 / 0.01 ms |
| Linear conv via FFT (zero-padded) | `numpy.convolve` | `1.9e-15` | 2.85 / 0.003 ms |
| Difference equation | `scipy.signal.lfilter` | `0` | 0.15 / 0.01 ms |
| Fourier filter (low-pass) | `numpy.fft` | `1.2e-15` | 0.96 / 0.03 ms |

**The worst error across all operations is about `7e-13`**, which is very close to the precision of floating-point numbers themselves (about `1e-16`). The difference comes from the order of additions and multiplications, not from an algorithmic mistake.

### 5.3 Speed

Our code is slower than numpy for anything involving the FFT (roughly 20 to 25 times on 1000 samples) and for the loop-based DTFT (roughly 400 times), because numpy is written in C and heavily optimized. This is expected and acceptable in an educational project, and we report it as a result in the discussion. For a 3-second audio signal (132,300 samples) our hand-written FFT takes about 0.3 seconds, which is fine for interactive use.

*Timings come from a single run on one machine and will differ on yours. Re-run the Accuracy tab and update these numbers before submitting.*

---

## 6. Tests

The whole project currently has **208 passing tests** (one is skipped on purpose because numpy swaps the argument order in one specific case). Created Signal coverage:

- **Waveforms:** peak value, period and shape for every type, and a match against scipy for every type at different frequencies and phases.
- **Noise:** measured SNR close to the requested one, and repeatability with the same seed.
- **Analysis:** peak at the right frequency and amplitude, cosine phase, and the aliasing effect.
- **Filters:** each filter removes the intended frequency and keeps the rest, and each matches numpy/scipy on both even and odd lengths.
- **DTFT:** agreement with the FFT, impulse and delayed impulse, cosine, symmetry, both peaks visible with noise, and rejection of oversized requests.
- **Convolution vs FFT:** agreement with `np.convolve`, and the circular error confined to the first `M−1` samples.
- **Difference equation:** agreement with `lfilter`, the impulse response `b·aⁿ`, the DC gain, and stability.
- **Comparison engine:** a test that makes sure the comparison **fails** when we deliberately break the code (otherwise it would say "match" forever).
- **GUI:** headless tests covering field visibility per filter, noise, aliasing and instability warnings, input errors, filling the comparison table, the DTFT and convolution tabs, CSV export, Reset, and that the phase plot always has points.

---

## 7. Problems we hit and how they were solved

| Problem | Cause | Fix |
|---|---|---|
| FFT very slow on 3 s of audio (about 8 s) | Length 132,300 splits down to a large odd piece that used a full DFT | **Bluestein (chirp-z)** for large odd lengths, plus a cached small DFT matrix for short pieces. Dropped to about 0.3 s |
| Possible fake mismatch in Square and Sawtooth at wave edges | Our formula differed in form from scipy's | Unified the formula using `θ mod 2π` |
| A bin exactly on the cutoff could be dropped | Floating-point rounding in the `f ≤ fc` test | A tiny relative tolerance (`1e-9`) in both the filter and the reference |
| Phase differed by `2π` from numpy at DC | `−π` and `+π` are the same angle | Circular error wrapped into `[−π, π]` |
| The phase spectrum sometimes looked empty | Weak bins were hidden, leaving isolated points, and matplotlib draws no line through a single point | Draw phase as dots, always keep the strongest bin, and add a threshold slider |
| Zero errors cannot be shown on a log axis | `log(0)` is undefined | A fixed floor of `1e-18`, in the plot only |
| The loop-based DTFT could freeze the UI | Work grows with samples × points | A 1.5 million operation cap with a clear error, and it only recomputes while its tab is open |
| The brief was not followed from the start (no DTFT, difference equation or zero-padding comparison) | The work was built from the initial plan | Reviewed the brief and added the missing parts |

---

## 8. Key design decisions (for the discussion)

- **The FFT is hand-written:** radix-2 Cooley-Tukey, halving the length while it is even, a DFT matrix for small pieces, and Bluestein for large odd ones. It works for any length without padding the signal.
- **The DTFT uses loops** (required by the brief). Other operations are allowed to use numpy/scipy, but we still wrote them by hand and then compared them with the libraries.
- **An independent reference:** the numpy/scipy code uses different library calls from ours.
- **Amplitude spectrum instead of the raw FFT:** so a peak equals the true amplitude of the component.
- **Computation separated from the UI:** all functions are free of PyQt, so they are easy to test and reuse.

---

## 9. Current limitations

- The GUI supports up to 3 components at once (the function itself supports more).
- Sampling rate is limited to the slider range (100 to 10,000 Hz) and duration to 5 seconds so live updates stay smooth.
- The Butterworth comparison uses the same mask definition on both sides, not `scipy.signal.butter`, because that is a different design (IIR) and any difference there would reflect design, not accuracy.
- The DTFT row in the Accuracy table only checks the first 128 samples because the loops are slow.
- Noise is off by default. To get the brief's exact signal (two sinusoids + noise), tick the Gaussian noise box.

---

## 10. Conclusion

The Created Signal module is complete and meets the brief for Students 1 and 2: a two-sinusoid signal with Gaussian noise, a loop-based DTFT with magnitude and phase, moving average through linear convolution, a comparison of time-domain convolution with frequency-domain multiplication that shows the role of zero-padding, and a first-order difference equation. The hand-written code matches the standard libraries to about `7e-13` in the worst case, is covered by automated tests, and is ready to connect with Speech and Image because the Core is shared.

---

