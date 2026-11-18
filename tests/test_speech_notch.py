"""Tests for the interference-tone / peak-detection / IIR-notch functions in
speech_signal/processing.py (course sheet: Student 3 = analysis, Student 4 = filtering)."""
import os

import numpy as np
import pytest
from scipy import signal as sps

from speech_signal import processing as sp

SAMPLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "assets", "speech_sample.wav")


def _amp(x, f, fs):
    """Amplitude of the component at f Hz (x must contain a whole number of periods or be long)."""
    X = np.abs(np.fft.rfft(x))
    return X[int(round(f * len(x) / fs))] / len(x) * 2


# ---------------------------------------------------------------- add_tone
def test_add_tone_adds_a_sine_of_the_right_frequency_and_size():
    fs = 8000
    y = sp.add_tone(np.zeros(fs), fs, 50, amplitude=0.2)
    assert len(y) == fs
    assert _amp(y, 50, fs) == pytest.approx(0.2, rel=1e-6)


def test_add_tone_clips_to_one():
    y = sp.add_tone(np.full(1000, 0.95), 8000, 60, amplitude=0.5)
    assert np.max(np.abs(y)) <= 1.0


@pytest.mark.parametrize("freq", [0, -5, 4000, 5000])
def test_add_tone_rejects_bad_frequency(freq):
    with pytest.raises(ValueError):
        sp.add_tone(np.zeros(100), 8000, freq)


def test_add_tone_rejects_negative_amplitude_and_bad_signal():
    with pytest.raises(ValueError):
        sp.add_tone(np.zeros(100), 8000, 50, amplitude=-1)
    with pytest.raises(ValueError):
        sp.add_tone(np.zeros((2, 5)), 8000, 50)


# ------------------------------------------------------- find_peak_frequency
@pytest.mark.parametrize("f0", [50, 60, 440])
def test_find_peak_frequency_finds_the_tone(f0):
    fs = 8000
    t = np.arange(2 * fs) / fs
    assert sp.find_peak_frequency(0.5 * np.sin(2 * np.pi * f0 * t), fs) == pytest.approx(f0, abs=0.5)


def test_find_peak_frequency_ignores_dc_and_respects_range():
    fs = 8000
    t = np.arange(fs) / fs
    x = 5.0 + 0.1 * np.sin(2 * np.pi * 50 * t) + 0.3 * np.sin(2 * np.pi * 1000 * t)
    assert sp.find_peak_frequency(x, fs) == pytest.approx(1000, abs=1)          # DC skipped
    assert sp.find_peak_frequency(x, fs, fmin=20, fmax=200) == pytest.approx(50, abs=1)


def test_find_peak_frequency_bad_range():
    with pytest.raises(ValueError):
        sp.find_peak_frequency(np.zeros(100), 8000, fmin=500, fmax=100)


def test_find_hum_in_real_speech():
    x, fs = sp.load_audio(SAMPLE)
    assert sp.find_peak_frequency(sp.add_tone(x, fs, 50, 0.1), fs) == pytest.approx(50, abs=0.5)


# ------------------------------------------------------- notch_coefficients
def test_notch_coefficients_structure():
    fs, f0 = 8000, 1000
    b, a = sp.notch_coefficients(f0, fs, bandwidth=100)
    w0 = 2 * np.pi * f0 / fs
    assert b == pytest.approx([1, -2 * np.cos(w0), 1])
    r = 1 - np.pi * 100 / fs
    assert a == pytest.approx([1, -2 * r * np.cos(w0), r * r])
    # zeros on the unit circle at +-w0, poles at radius r (inside -> stable)
    assert np.abs(np.roots(b)) == pytest.approx([1, 1])
    assert np.abs(np.roots(a)) == pytest.approx([r, r])
    assert np.all(np.abs(np.roots(a)) < 1)


def test_notch_coefficients_explicit_r_overrides_bandwidth():
    _, a = sp.notch_coefficients(1000, 8000, r=0.9, bandwidth=1)
    assert a[2] == pytest.approx(0.81)


def test_notch_gain_is_zero_at_f0_and_one_elsewhere():
    fs = 44100
    b, a = sp.notch_coefficients(50, fs, bandwidth=10)
    f, h = sps.freqz(b, a, worN=[50, 440, 5000], fs=fs)
    assert abs(h[0]) < 1e-9
    assert abs(h[1]) == pytest.approx(1, abs=0.01)
    assert abs(h[2]) == pytest.approx(1, abs=1e-3)


@pytest.mark.parametrize("kwargs", [dict(f0=0), dict(f0=4000), dict(f0=-1), dict(f0=100, r=1.0),
                                    dict(f0=100, r=0), dict(f0=100, bandwidth=0),
                                    dict(f0=100, bandwidth=1e6)])
def test_notch_coefficients_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        sp.notch_coefficients(fs=8000, **kwargs)


# --------------------------------------------------------------- apply_notch
def test_apply_notch_matches_scipy_lfilter():
    fs = 44100
    x = 0.05 * np.random.default_rng(0).standard_normal(20000)       # small: no clipping
    b, a = sp.notch_coefficients(60, fs, bandwidth=5)
    assert sp.apply_notch(x, fs, 60, bandwidth=5) == pytest.approx(sps.lfilter(b, a, x), abs=1e-10)


def test_apply_notch_removes_only_the_hum():
    fs = 44100
    t = np.arange(4 * fs) / fs
    x = 0.2 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 50 * t)
    y = sp.apply_notch(x, fs, 50)
    tail = y[fs:]                                   # skip the start-up transient
    assert _amp(tail, 50, fs) < 0.2 * 0.01         # hum down by > 40 dB
    assert _amp(tail, 440, fs) == pytest.approx(0.2, rel=0.02)


def test_hum_removal_on_real_speech_lowers_the_error():
    clean, fs = sp.load_audio(SAMPLE)
    noisy = sp.add_tone(clean, fs, 50, 0.1)
    filtered = sp.apply_notch(noisy, fs, 50)
    mse = lambda a, b: float(np.mean((a - b) ** 2))
    assert len(filtered) == len(clean)
    assert mse(filtered, clean) < 0.05 * mse(noisy, clean)


def test_apply_notch_output_range_and_bad_input():
    y = sp.apply_notch(np.full(2000, 0.99), 8000, 100)
    assert np.max(np.abs(y)) <= 1.0
    with pytest.raises(ValueError):
        sp.apply_notch(np.zeros(100), 8000, 5000)
    with pytest.raises(ValueError):
        sp.apply_notch(np.array([0.1]), 8000, 100)
