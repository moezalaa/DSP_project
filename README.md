# DSP Project: Signal Processing Toolkit

A Python desktop application for exploring Digital Signal Processing concepts on three types of signals:

1. **Created Signal**: synthetic 1D signals generated from parameters
2. **Image**: 2D signals (grayscale images)
3. **Speech**: recorded or loaded audio

Each signal type has its own GUI tab where you can apply filters (convolution, Fourier-based filtering, and more) and visualize the **magnitude** and **phase** spectra.

---

## Features

### Created Signal
- Generate Sine, Cosine, Square, Triangle, and Sawtooth waves (and sums of several)
- Control amplitude, frequency, phase, sampling rate, and duration
- Add Gaussian noise
- Apply convolution (moving average, custom kernel) and frequency-domain filters (LPF / HPF / BPF)
- Plot time domain, magnitude spectrum, and phase spectrum

### Image
- Load an image and convert it to grayscale
- 2D convolution: blur (box/Gaussian), sharpen, Sobel, Prewitt, Laplacian
- 2D FFT with `fftshift`, magnitude (log scale) and phase display
- Frequency-domain filters: Ideal / Gaussian / Butterworth, low-pass and high-pass

### Speech
- Record from the microphone or load a WAV file
- Play original and filtered audio
- FFT magnitude/phase and spectrogram (STFT)
- Filters: LPF / HPF / BPF, noise reduction, echo (convolution with an impulse response)
- Save the filtered audio

---

## Project Structure

```
dsp_project/
├── main.py                  # Main window with 3 tabs
├── core/                    # Shared algorithms
│   ├── fourier.py           # DFT / FFT / IFFT
│   ├── convolution.py       # 1D & 2D convolution
│   ├── filters.py           # LPF, HPF, BPF, Notch, ...
│   └── utils.py             # magnitude, phase, normalization
├── created_signal/          # Created Signal module + GUI
├── image_signal/            # Image module + GUI
├── speech_signal/           # Speech module + GUI
├── tests/                   # Compare our implementations against numpy/scipy
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd dsp_project

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### `requirements.txt`

```
numpy
scipy
matplotlib
PyQt5
opencv-python
Pillow
soundfile
sounddevice
```

---

## Usage

```bash
python main.py
```

Pick a tab (**Created Signal**, **Image**, or **Speech**), load or generate a signal, choose a filter, set its parameters, and view the result together with its magnitude and phase.

---

## Core API Convention

All filters follow the same interface so any GUI can call any core module:

```python
output, magnitude, phase = apply_filter(signal, filter_type, **params)
```

---

## Testing

Our hand-written DFT, FFT, convolution and filters are verified against `numpy` / `scipy`:

```bash
python -m pytest tests/        # 200+ tests, takes a few seconds
```

## Status

| Part | Status |
|---|---|
| `core/` (utils, fourier, convolution, filters, gui_widgets) | Done (Day 1-2) |
| `main.py` (3 tabs, auto-loads each team's tab) | Done (Day 1-2) |
| Created Signal (incl. DTFT loops, conv vs FFT, difference equation, numpy/scipy comparison, GUI) | Done |
| Image / Speech | Next |

## How a team plugs its tab into `main.py`

Create the class in your package and `main.py` picks it up automatically:

| File | Class |
|---|---|
| `created_signal/gui.py` | `CreatedSignalTab(QWidget)` |
| `image_signal/gui.py` | `ImageSignalTab(QWidget)` |
| `speech_signal/gui.py` | `SpeechSignalTab(QWidget)` |

---

## Team

| Module | Members |
|---|---|
| Created Signal | Member 1, Member 2 |
| Image | Member 3, Member 4 |
| Speech | Member 5, Member 6 |

---

## Git Workflow

- One branch per module: `feature/created-signal`, `feature/image`, `feature/speech`
- Merge into `main` only through a Pull Request reviewed by another team
- Keep shared code in `core/` and avoid duplicating it inside modules

---

## Course

Digital Signal Processing: <University / Faculty name>, <Academic year>
Supervised by: <Dr. / Eng. name>
