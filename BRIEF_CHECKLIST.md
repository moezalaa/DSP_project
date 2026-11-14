# Checklist against the project brief

The brief splits the work into 6 student roles. Our three teams cover them like this:

| Team | Brief roles |
|---|---|
| Created Signal (Moaz, Omar) | Omar (Mathematical signal & DTFT) + Moez(Convolution & difference equations) |
| Speech (Fatma, Ghada) | Student 3 (Audio analysis) + Student 4 (Audio filtering) |
| Image (Mohab, Hagar) | Student 5 (2D Fourier & image filtering) + Student 6 (Image noise removal & evaluation) |

## Created Signal: DONE

### Omar: Mathematical Signal & DTFT
| Brief requirement | Where |
|---|---|
| Signal with two sinusoids + Gaussian noise | Components 1 and 2 (default 5 Hz + 40 Hz) + "Gaussian noise" box |
| DTFT from its defining equation, using loops, limited samples and frequency points | `processing.dtft` (double loop, no FFT), "DTFT (loops)" tab with N and number of points sliders |
| Magnitude and phase spectra | "DTFT (loops)" tab (w in rad/sample) and "Signals" tab (Hz) |
| Theory: angular frequency, DTFT, magnitude, phase | Info line on the DTFT tab: w = 2*pi*f/fs, w = pi is Nyquist |

### Moez: Convolution & Difference Equations
| Brief requirement | Where |
|---|---|
| Moving average with linear convolution | Filter "Moving average (convolution)" (`core.convolution.convolve1d`) |
| Compare time-domain convolution with frequency-domain multiplication, zero-padding | "Convolution vs FFT" tab: zero-padded matches (~1e-15), no padding = circular (error in the first M-1 samples) |
| Simple first-order difference equation | Filter "Difference equation: y[n] = a*y[n-1] + b*x[n]" (`first_order_difference_equation`), with a stability warning when |a| >= 1 |

### Extra (not required by the brief)
Everything is also compared against numpy/scipy in the "Accuracy" tab .


