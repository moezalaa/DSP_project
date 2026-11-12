"""Run the whole Speech core on a real recording (no GUI needed).

    python scripts/speech_demo.py                      # uses assets/speech_sample.wav
    python scripts/speech_demo.py path/to/your.wav     # or your own file

It prints a short summary, writes figures to docs/figures/ (for the report) and
writes listenable results to outputs/ (original, filtered, echo, denoised).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from speech_signal import processing as sp  # noqa: E402

FIG = os.path.join(ROOT, "docs", "figures")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(FIG, exist_ok=True)
os.makedirs(OUT, exist_ok=True)


def snr_db(clean, noisy):
    return 10 * np.log10(np.sum(clean ** 2) / np.sum((noisy - clean) ** 2))


def spectrogram_plot(ax, x, fs, title, fmax=8000):
    t, f, S = sp.compute_spectrogram(x, fs)
    keep = f <= fmax
    ax.pcolormesh(t, f[keep], S[keep], shading="auto", vmin=S.max() - 90, vmax=S.max(), cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "assets", "speech_sample.wav")
    x, fs = sp.load_audio(path)
    t = np.arange(len(x)) / fs
    print(f"Loaded {path}: {len(x)} samples, fs = {fs} Hz, {len(x) / fs:.2f} s")

    # 1) waveform + spectrum + phase -------------------------------------------------
    f, mag, ph = sp.compute_fft(x, fs)
    fig, ax = plt.subplots(3, 1, figsize=(9, 8))
    ax[0].plot(t, x, lw=0.5); ax[0].set_title("Waveform"); ax[0].set_xlabel("Time (s)")
    ax[1].plot(f, mag, lw=0.6); ax[1].set_xlim(0, 8000); ax[1].set_title("Magnitude spectrum")
    ax[1].set_xlabel("Frequency (Hz)")
    ax[2].plot(f, ph, lw=0.4); ax[2].set_xlim(0, 8000); ax[2].set_title("Phase spectrum")
    ax[2].set_xlabel("Frequency (Hz)"); ax[2].set_ylabel("rad")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "speech_1_fft.png"), dpi=120); plt.close(fig)
    print(f"Strongest frequency: {f[np.argmax(mag)]:.0f} Hz")

    # 2) spectrogram -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4))
    spectrogram_plot(ax, x, fs, "Spectrogram (Hamming 1024, hop 512)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "speech_2_spectrogram.png"), dpi=120); plt.close(fig)

    # 3) filters ---------------------------------------------------------------------
    cases = [("Low-pass 1000 Hz", "lowpass", 1000), ("High-pass 1000 Hz", "highpass", 1000),
             ("Band-pass 300-3400 Hz (telephone)", "bandpass", (300, 3400))]
    fig, ax = plt.subplots(len(cases), 1, figsize=(9, 8))
    for a, (name, kind, cut) in zip(ax, cases):
        y = sp.filter_audio(x, fs, kind, cut)
        fy, my, _ = sp.compute_fft(y, fs)
        a.plot(f, mag, lw=0.5, alpha=0.5, label="original")
        a.plot(fy, my, lw=0.6, label="filtered")
        a.set_xlim(0, 8000); a.set_title(name); a.set_xlabel("Frequency (Hz)"); a.legend(loc="upper right")
        sp.save_audio(os.path.join(OUT, f"filtered_{kind}.wav"), y, fs)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "speech_3_filters.png"), dpi=120); plt.close(fig)

    # 4) echo ------------------------------------------------------------------------
    e = sp.add_echo(x, fs, 0.25, 0.5, n_echoes=2)
    te = np.arange(len(e)) / fs
    fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    ax[0].plot(t, x, lw=0.5); ax[0].set_title("Original")
    ax[1].plot(te, e, lw=0.5); ax[1].set_title("With echo (delay 0.25 s, decay 0.5, 2 repeats)")
    ax[1].set_xlabel("Time (s)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "speech_4_echo.png"), dpi=120); plt.close(fig)
    sp.save_audio(os.path.join(OUT, "echo.wav"), e, fs)

    # 5) noise reduction -------------------------------------------------------------
    rng = np.random.default_rng(0)
    noisy = x + 0.03 * rng.standard_normal(len(x))
    clean = sp.reduce_noise(noisy, fs)
    print(f"Noise reduction: SNR {snr_db(x, noisy):.1f} dB -> {snr_db(x, clean):.1f} dB")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    spectrogram_plot(ax[0], x, fs, "Original")
    spectrogram_plot(ax[1], noisy, fs, f"Noisy (SNR {snr_db(x, noisy):.1f} dB)")
    spectrogram_plot(ax[2], clean, fs, f"After spectral subtraction (SNR {snr_db(x, clean):.1f} dB)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "speech_5_noise.png"), dpi=120); plt.close(fig)
    sp.save_audio(os.path.join(OUT, "noisy.wav"), noisy, fs)
    sp.save_audio(os.path.join(OUT, "denoised.wav"), clean, fs)
    sp.save_audio(os.path.join(OUT, "original.wav"), x, fs)

    print(f"Figures -> {FIG}\nAudio   -> {OUT}")


if __name__ == "__main__":
    main()
