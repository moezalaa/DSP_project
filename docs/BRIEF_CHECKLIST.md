# Checklist against the project brief

The brief splits the work into 6 student roles. Our three teams cover them like this:

| Team | Brief roles |
|---|---|
| Created Signal (Moez, Omar) | Omar (Mathematical signal & DTFT) + Moez(Convolution & difference equations) |
| Speech (Fatma, Ghada) | Ghada (Audio analysis) + Fatma (Audio filtering) |
| Image (Mohab, Hagar) | Hager (2D Fourier & image filtering) + Mohab (Image noise removal & evaluation) |

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

### Extra (not required by the brief) (Moez)
Everything is also compared against numpy/scipy in the "Accuracy" tab .

## Speech: DONE (Fatma = core, Ghada = GUI), verified by tests

### Ghada: Audio Analysis
| Brief requirement | Where |
|---|---|
| Load a WAV file | `processing.load_audio` / "Load audio file" (stereo -> mono, errors reported) |
| Add a 50 or 60 Hz interference tone | `processing.add_tone` / "Interference tone" box with 50 Hz and 60 Hz buttons |
| Clean and noisy waveforms and spectra | "Time & Spectrum" tab, "Compare with: Original" overlay |
| Identify the interference frequency | `processing.find_peak_frequency` / "Find strongest frequency" (red marker on the spectrum) |

### Fatma: Audio Filtering
| Brief requirement | Where |
|---|---|
| Second-order notch filter, coefficients calculated for the chosen frequency | `processing.notch_coefficients` (b = [1, -2cos w0, 1], a = [1, -2r cos w0, r^2], r = 1 - pi*bw/fs); the GUI prints b and a |
| Process the audio | `processing.apply_notch` (difference equation, sample by sample) |
| Save the filtered WAV | `processing.save_audio` / "Save processed" |
| Compare spectra and listening results | Compare overlay, Spectrogram tab, "Play Original / Processed" |

Extras the team added: low/high/band-pass, echo (convolution), spectral-subtraction noise reduction, spectrogram.


## Image (Hager = core, Mohab = GUI): DONE

### Hager: 2D Fourier & Image Filtering
| Brief requirement | Where |
|---|---|
| Load an image and convert it to grayscale | `processing.load_image` (RGB, RGBA, palette, 1-bit, 16-bit and float files) |
| 2D Fourier spectrum with fft2 and fftshift | `processing.image_fft` / "Analyze FFT" (log magnitude + phase) |
| Ideal circular low-pass and high-pass masks, filtered images shown | `frequency_filter(kind="ideal")` / "Frequency filter" box (Gaussian and Butterworth too) |

### Mohab: Image Noise Removal & Evaluation
| Brief requirement | Where |
|---|---|
| Periodic sinusoidal noise | `denoising.add_periodic_noise` (frequency must be below half the image size) |
| Noisy image spectrum | shown by "Add Periodic Noise" and again next to the restored spectrum |
| Notch mask from the known noise frequency | `denoising.create_notch_mask`, displayed in the third column |
| Inverse 2D FFT; compare original, noisy and restored | "Remove Periodic Noise": original / restored / noisy side by side, with both spectra and the mask |
| MSE and PSNR | `image_signal/metrics.py`; the status line shows MSE, RMSE and PSNR, before and after |

Run `python scripts/check_brief.py` to re-check all 19 programming tasks of the brief on the real code.
 



