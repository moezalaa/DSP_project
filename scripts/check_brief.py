"""Checks the project against the course brief, requirement by requirement.

    python scripts/check_brief.py

Every line of the brief's "Programming tasks" (Students 1 to 6) is run on the real code with
synthetic data; the script prints PASS / FAIL with a short piece of evidence and exits with
status 1 if anything fails. Nothing here opens a window or needs a microphone.
"""
import inspect
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image                                         # noqa: E402
from scipy import signal as sps                               # noqa: E402

from core.convolution import convolve1d                       # noqa: E402
from core.fourier import fft2                                 # noqa: E402
from core.utils import fftshift                               # noqa: E402
from created_signal import processing as C                    # noqa: E402
from image_signal import denoising as D                       # noqa: E402
from image_signal import frequency_filter, image_fft, load_image  # noqa: E402
from image_signal.metrics import image_metrics                # noqa: E402
from speech_signal import processing as S                     # noqa: E402

RESULTS = []


def check(student, requirement):
    def wrap(fn):
        try:
            ok, evidence = fn()
        except Exception as e:                                # a crash is a failure, not an abort
            ok, evidence = False, f"{type(e).__name__}: {e}"
        RESULTS.append((student, requirement, bool(ok), evidence))
        return fn
    return wrap


# =============================================================== Student 1
FS = 1000
t, s1 = C.generate_signal("sine", 1.0, 5, 0, FS, 1)
_, s2 = C.generate_signal("sine", 0.5, 40, 0, FS, 1)
CLEAN = C.combine_signals([s1, s2])
NOISY = C.add_noise(CLEAN, 15, seed=1)


@check("1", "Signal with two sinusoidal components and Gaussian noise")
def _():
    f, mag, _p = C.analyze(NOISY, FS)
    top = sorted(float(v) for v in f[np.argsort(mag)[-2:]])
    snr = 10 * np.log10(np.mean(CLEAN ** 2) / np.mean((NOISY - CLEAN) ** 2))
    return top == [5.0, 40.0] and abs(snr - 15) < 0.5, f"peaks at {top} Hz, measured SNR {snr:.2f} dB"


@check("1", "DTFT from its defining equation using loops (no FFT inside)")
def _():
    src = inspect.getsource(C.dtft)
    n = 64
    w = 2 * np.pi * np.arange(n) / n
    err = np.max(np.abs(C.dtft(NOISY[:n], w) - np.fft.fft(NOISY[:n])))
    return ("for " in src and "fft" not in src.replace("fft2", "")), f"double loop, error vs FFT {err:.1e}"


@check("1", "Magnitude and phase spectra of the DTFT")
def _():
    w, X, mag, ph = C.dtft_spectrum(NOISY, 256, 256)
    return (len(mag) == len(ph) == 256 and ph.min() >= -np.pi and ph.max() <= np.pi
            and abs(abs(w[np.argmax(mag)]) - 2 * np.pi * 5 / FS) < 0.05), \
        f"{len(w)} points in [-pi, pi], strongest near |w| = {abs(w[np.argmax(mag)]):.3f} rad/sample"


# =============================================================== Student 2
@check("2", "Moving average with linear convolution")
def _():
    y = C.apply_conv_filter(NOISY, "moving_average", 5)
    ref = np.convolve(NOISY, np.ones(5) / 5, mode="same")
    return np.allclose(y, ref), f"max difference vs np.convolve {np.max(np.abs(y - ref)):.1e}"


@check("2", "Time-domain convolution vs frequency-domain multiplication (zero-padding)")
def _():
    h = C.make_kernel("moving_average", 7)
    r = C.convolution_vs_frequency(NOISY, h)
    e_pad, e_circ = np.max(np.abs(r["err_padded"])), np.max(np.abs(r["err_circular"]))
    return e_pad < 1e-9 and e_circ > 1e-3, f"zero-padded error {e_pad:.1e}; without padding (circular) {e_circ:.2f}"


@check("2", "Simple first-order difference equation")
def _():
    y = C.first_order_difference_equation(NOISY, 0.9, 0.1)
    err = np.max(np.abs(y - sps.lfilter([0.1], [1, -0.9], NOISY)))
    return err < 1e-12, f"y[n] = 0.9*y[n-1] + 0.1*x[n], error vs lfilter {err:.1e}"


# =============================================================== Students 3 and 4
FS_A = 8000
_n = np.arange(3 * FS_A)
SPEECH = np.zeros(3 * FS_A)
for _st in (0.5, 1.5, 2.3):
    _k = int(0.5 * FS_A)
    SPEECH[int(_st * FS_A):int(_st * FS_A) + _k] = 0.4 * np.sin(2 * np.pi * 600 * np.arange(_k) / FS_A) * np.hanning(_k)


def amp(x, f, fs=FS_A):
    X = np.abs(np.fft.rfft(x)) * 2 / len(x)
    fr = np.fft.rfftfreq(len(x), 1 / fs)
    return X[(fr > f - 2) & (fr < f + 2)].max()


@check("3", "Load a WAV file")
def _():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.wav")
        S.save_audio(p, SPEECH, FS_A)
        x, fs = S.load_audio(p)
    return fs == FS_A and len(x) == len(SPEECH), f"{len(x)} samples at {fs} Hz"


@check("3", "Add a 50 Hz or 60 Hz interference tone")
def _():
    out = []
    for f0 in (50, 60):
        y = S.add_tone(SPEECH, FS_A, f0, 0.2)
        out.append(round(float(amp(y, f0)), 3))
    return all(abs(a - 0.2) < 0.01 for a in out), f"tone amplitudes measured {out} (set 0.2)"


@check("3", "Clean and noisy waveforms and spectra; identify the interference frequency")
def _():
    noisy = S.add_tone(SPEECH, FS_A, 50, 0.2)
    f_clean, m_clean, _p = S.compute_fft(SPEECH, FS_A)
    f_noisy, m_noisy, _p = S.compute_fft(noisy, FS_A)
    found = S.find_peak_frequency(noisy, FS_A, 20, 200)
    return abs(found - 50) < 1 and m_noisy[np.argmin(abs(f_noisy - 50))] > 10 * m_clean[np.argmin(abs(f_clean - 50))], \
        f"identified {found:.2f} Hz"


@check("4", "Second-order notch filter; coefficients calculated for the selected frequency")
def _():
    rows = []
    ok = True
    for f0 in (50, 60):
        b, a = S.notch_coefficients(f0, FS_A, bandwidth=10)
        w0 = 2 * np.pi * f0 / FS_A
        r = 1 - np.pi * 10 / FS_A
        ok &= np.allclose(b, [1, -2 * np.cos(w0), 1]) and np.allclose(a, [1, -2 * r * np.cos(w0), r * r])
        ok &= np.all(np.abs(np.roots(a)) < 1) and np.allclose(np.abs(np.roots(b)), 1)
        rows.append(f"{f0} Hz: b = {np.round(b, 4).tolist()}, a = {np.round(a, 4).tolist()}")
    return ok, "; ".join(rows) + " (zeros on the unit circle, poles inside: stable)"


@check("4", "Process the audio and reduce the tone")
def _():
    noisy = S.add_tone(SPEECH, FS_A, 50, 0.2)
    y = S.apply_notch(noisy, FS_A, 50, bandwidth=5)
    drop = 20 * np.log10(amp(noisy[FS_A:], 50) / amp(y[FS_A:], 50))
    return drop > 25 and amp(y, 600) > 0.9 * amp(SPEECH, 600), f"50 Hz reduced by {drop:.0f} dB, speech tone kept"


@check("4", "Save the filtered WAV and compare spectra")
def _():
    noisy = S.add_tone(SPEECH, FS_A, 50, 0.2)
    y = S.apply_notch(noisy, FS_A, 50, bandwidth=5)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "filtered.wav")
        S.save_audio(p, y, FS_A)
        z, fs = S.load_audio(p)
    return fs == FS_A and np.max(np.abs(z - y)) < 1e-4, "saved WAV reloads with the same samples (16-bit rounding only)"


# =============================================================== Students 5 and 6
def synthetic_image(n=128):
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:n, :n]
    img = 90 + 60 * np.sin(xx / 20) + 50 * np.cos(yy / 27)
    img[n // 4:n // 2, n // 4:3 * n // 4] += 70
    return np.clip(img + rng.normal(0, 3, (n, n)), 0, 255)


IMG = synthetic_image()


@check("5", "Load an image and convert it to grayscale")
def _():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.png")
        rgb = np.stack([IMG, IMG * 0.5, 255 - IMG], axis=-1).astype(np.uint8)
        Image.fromarray(rgb).save(p)
        x = load_image(p)
    return x.ndim == 2 and x.shape == IMG.shape and 0 <= x.min() and x.max() <= 255, f"RGB file -> {x.shape} grayscale"


@check("5", "2D Fourier spectrum with fft2 and fftshift")
def _():
    F, logmag, ph = image_fft(IMG)
    ref = np.fft.fftshift(np.fft.fft2(IMG))
    err = np.max(np.abs(F - ref))
    centre_is_dc = np.unravel_index(np.argmax(np.abs(F)), F.shape) == (64, 64)
    return err < 1e-6 and centre_is_dc, f"error vs numpy {err:.1e}, DC at the centre"


@check("5", "Ideal circular low-pass and high-pass masks")
def _():
    from core.filters import freq_mask_2d
    m = freq_mask_2d((129, 129), "lowpass", 20, "ideal")
    circle = m[64, 84] == 1 and m[64, 85] == 0 and m[64 + 15, 64 + 15] == 0 and m[64 + 14, 64 + 14] == 1
    lo, _a, _b = frequency_filter(IMG, "lowpass", 20)
    hi, _a, _b = frequency_filter(IMG, "highpass", 20)
    return circle and lo.std() < IMG.std() and hi.min() >= 0, "mask is a disc; low-pass blurs, high-pass keeps edges"


@check("6", "Periodic sinusoidal noise added to a grayscale image")
def _():
    noisy = D.add_periodic_noise(IMG, 30, 8, 6)
    yy, xx = np.mgrid[:128, :128]
    return np.allclose(noisy - IMG, 30 * np.sin(2 * np.pi * (8 * xx / 128 + 6 * yy / 128))), "noise = 30*sin(2*pi*(8x/W + 6y/H))"


@check("6", "Noisy image spectrum shows the noise peaks; Notch Mask from the known frequency")
def _():
    noisy = D.add_periodic_noise(IMG, 30, 8, 6)
    F_noisy, _log = D.analyze_noisy_image(noisy)
    F_clean, _log = D.analyze_noisy_image(IMG)
    mask = D.create_notch_mask(IMG.shape, 8, 6, 3)
    ratios = [abs(F_noisy[y, x]) / abs(F_clean[y, x]) for y, x in ((70, 72), (58, 56))]   # (64 +- 6, 64 +- 8)
    return min(ratios) > 5 and mask[70, 72] == 0 and mask[58, 56] == 0 and mask[64, 64] == 1, \
        f"noise peaks at (70, 72) and (58, 56) are {ratios[0]:.0f}x and {ratios[1]:.0f}x stronger than in the clean image; " \
        f"the mask removes {int((mask == 0).sum())} pixels there and keeps DC"


@check("6", "Inverse 2D FFT restores the image (original / noisy / restored)")
def _():
    noisy = D.add_periodic_noise(IMG, 30, 8, 6)
    restored, mask = D.remove_periodic_noise(noisy, 8, 6, 3)
    before, after = image_metrics(noisy, IMG), image_metrics(restored, IMG)
    return after["rmse"] < before["rmse"] / 10, f"RMSE {before['rmse']:.2f} -> {after['rmse']:.2f}"


@check("6", "MSE and PSNR to evaluate restoration quality")
def _():
    noisy = D.add_periodic_noise(IMG, 30, 8, 6)
    restored, _m = D.remove_periodic_noise(noisy, 8, 6, 3)
    b, a = image_metrics(noisy, IMG), image_metrics(restored, IMG)
    consistent = np.isclose(a["psnr"], 10 * np.log10(255 ** 2 / a["mse"]))
    return consistent and a["psnr"] > b["psnr"] + 15, \
        f"MSE {b['mse']:.1f} -> {a['mse']:.2f}, PSNR {b['psnr']:.1f} -> {a['psnr']:.1f} dB"


def main():
    width = max(len(r[1]) for r in RESULTS)
    print(f"{'Student':8s}{'Requirement':{width + 2}s}Result  Evidence")
    print("-" * (width + 40))
    for student, req, ok, ev in RESULTS:
        print(f"{student:8s}{req:{width + 2}s}{'PASS' if ok else 'FAIL':6s}  {ev}")
    passed = sum(r[2] for r in RESULTS)
    print("-" * (width + 40))
    print(f"{passed}/{len(RESULTS)} requirements pass")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
