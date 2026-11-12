"""Tests for speech_signal/processing.py (owner: Fatma).

Real-audio tests use assets/speech_sample.wav (5 s of real speech, 44.1 kHz).
No microphone is needed: record_audio is tested with a fake `sounddevice`.
"""
import os
import sys
import types

import numpy as np
import pytest
import soundfile as sf
from scipy import signal as sps

from speech_signal import processing as sp

ASSET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "assets", "speech_sample.wav")
FS = 8000
T = np.arange(FS) / FS            # 1 second


def tone(f, fs=FS, n=FS):
    return np.sin(2 * np.pi * f * np.arange(n) / fs)


def snr_db(clean, noisy):
    return 10 * np.log10(np.sum(clean ** 2) / np.sum((noisy - clean) ** 2))


@pytest.fixture(scope="module")
def speech():
    x, fs = sp.load_audio(ASSET)
    return x, fs


# ----------------------------------------------------------------------------
# load_audio / save_audio
# ----------------------------------------------------------------------------
def test_load_real_audio(speech):
    x, fs = speech
    assert fs == 44100
    assert x.ndim == 1 and x.dtype == np.float64
    assert len(x) == 5 * 44100
    assert -1.0 <= x.min() and x.max() <= 1.0
    assert x.std() > 0.01                       # it is not silence


def test_save_load_round_trip(tmp_path):
    x = 0.5 * tone(440)
    path = tmp_path / "t.wav"
    sp.save_audio(path, x, FS)
    y, fs = sp.load_audio(path)
    assert fs == FS and len(y) == len(x)
    assert np.allclose(x, y, atol=1 / 32768)    # 16-bit quantisation only


def test_save_clips_out_of_range(tmp_path):
    path = tmp_path / "loud.wav"
    sp.save_audio(path, np.array([0.0, 3.0, -3.0, 0.5]), FS)
    y, _ = sp.load_audio(path)
    assert y.max() <= 1.0 and y.min() >= -1.0


def test_stereo_is_averaged_to_mono(tmp_path):
    left, right = 0.4 * tone(100), 0.2 * tone(100)
    path = tmp_path / "st.wav"
    sf.write(path, np.column_stack([left, right]), FS, subtype="FLOAT")
    y, _ = sp.load_audio(path)
    assert y.ndim == 1 and np.allclose(y, 0.3 * tone(100), atol=1e-6)


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(ValueError):
        sp.load_audio(tmp_path / "nope.wav")


def test_load_corrupt_file_raises(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"this is not a wav file at all")
    with pytest.raises(ValueError):
        sp.load_audio(bad)


def test_load_empty_audio_raises(tmp_path):
    path = tmp_path / "empty.wav"
    sf.write(path, np.zeros(0), FS)
    with pytest.raises(ValueError):
        sp.load_audio(path)


def test_save_rejects_bad_input(tmp_path):
    with pytest.raises(ValueError):
        sp.save_audio(tmp_path / "a.wav", np.array([]), FS)
    with pytest.raises(ValueError):
        sp.save_audio(tmp_path / "a.wav", tone(10), 0)
    with pytest.raises(ValueError):
        sp.save_audio(tmp_path / "no_such_dir" / "a.wav", tone(10), FS)


# ----------------------------------------------------------------------------
# record_audio (fake microphone)
# ----------------------------------------------------------------------------
def _fake_sounddevice(monkeypatch, fail=False):
    fake = types.ModuleType("sounddevice")

    def rec(frames, samplerate, channels, dtype):
        if fail:
            raise RuntimeError("no input device")
        return np.full((frames, channels), 0.25)

    fake.rec = rec
    fake.wait = lambda: None
    monkeypatch.setitem(sys.modules, "sounddevice", fake)


def test_record_audio_shape_and_range(monkeypatch):
    _fake_sounddevice(monkeypatch)
    x = sp.record_audio(0.5, fs=8000)
    assert x.shape == (4000,) and x.dtype == np.float64
    assert np.allclose(x, 0.25)


def test_record_audio_bad_arguments(monkeypatch):
    _fake_sounddevice(monkeypatch)
    for dur in (0, -1, 10_000):
        with pytest.raises(ValueError):
            sp.record_audio(dur)
    with pytest.raises(ValueError):
        sp.record_audio(1, fs=0)


def test_record_audio_without_microphone_raises_valueerror(monkeypatch):
    _fake_sounddevice(monkeypatch, fail=True)
    with pytest.raises(ValueError, match="microphone"):
        sp.record_audio(1)


# ----------------------------------------------------------------------------
# compute_fft
# ----------------------------------------------------------------------------
def test_compute_fft_matches_numpy():
    x = np.random.default_rng(0).standard_normal(1001)       # odd length on purpose
    f, m, p = sp.compute_fft(x, FS)
    ref = np.fft.rfft(x)
    assert np.allclose(f, np.fft.rfftfreq(len(x), 1 / FS), atol=1e-9)
    assert np.allclose(m, np.abs(ref), atol=1e-6)
    assert np.allclose(np.cos(p), np.cos(np.angle(ref)), atol=1e-6)   # compare angles safely


def test_compute_fft_peak_and_ranges():
    f, m, p = sp.compute_fft(tone(440), FS)
    assert f[0] == 0 and f[-1] == FS / 2
    assert f[np.argmax(m)] == pytest.approx(440)
    assert p.min() >= -np.pi and p.max() <= np.pi
    assert len(f) == len(m) == len(p) == FS // 2 + 1


def test_compute_fft_real_speech(speech):
    x, fs = speech
    f, m, _ = sp.compute_fft(x, fs)
    assert np.allclose(m, np.abs(np.fft.rfft(x)), atol=1e-6)
    low = m[f < 4000].sum()
    assert low > 0.5 * m.sum()                  # speech energy sits at low frequencies


@pytest.mark.parametrize("bad", [np.array([]), np.array([1.0]), np.zeros((4, 4)),
                                 np.array([1.0, np.nan, 2.0])])
def test_compute_fft_bad_signal(bad):
    with pytest.raises(ValueError):
        sp.compute_fft(bad, FS)


def test_compute_fft_bad_fs():
    with pytest.raises(ValueError):
        sp.compute_fft(tone(10), -1)


# ----------------------------------------------------------------------------
# compute_spectrogram
# ----------------------------------------------------------------------------
def test_spectrogram_matches_numpy_frames():
    x = np.random.default_rng(1).standard_normal(5000)
    win, hop = 256, 100
    times, freqs, S = sp.compute_spectrogram(x, FS, window_size=win, hop=hop)
    w = np.hamming(win)
    n_frames = 1 + (len(x) - win) // hop
    assert S.shape == (win // 2 + 1, n_frames) == (len(freqs), len(times))
    for m in (0, 7, n_frames - 1):
        ref = 20 * np.log10(np.abs(np.fft.rfft(x[m * hop:m * hop + win] * w)) + 1e-12)
        assert np.allclose(S[:, m], ref, atol=1e-6)
    assert np.allclose(freqs, np.fft.rfftfreq(win, 1 / FS))
    assert times[0] == pytest.approx((win / 2) / FS)
    assert np.allclose(np.diff(times), hop / FS)


def test_spectrogram_close_to_scipy():
    """Scipy's spectrogram uses a different scaling, but the SHAPE must agree."""
    x = tone(1000) + 0.5 * tone(2500)
    _, freqs, S = sp.compute_spectrogram(x, FS, 512, 256)
    f_s, _, Sxx = sps.spectrogram(x, FS, window="hamming", nperseg=512, noverlap=256,
                                  mode="magnitude", scaling="spectrum")
    assert np.allclose(freqs, f_s)
    mine = S[:, 3] - S[:, 3].max()
    ref = 20 * np.log10(Sxx[:, 3] + 1e-12)
    ref -= ref.max()
    near_peaks = (mine > -40)
    assert np.allclose(mine[near_peaks], ref[near_peaks], atol=0.5)


def test_spectrogram_tracks_a_tone():
    _, freqs, S = sp.compute_spectrogram(tone(1000), FS, 512, 256)
    assert np.all(np.abs(freqs[np.argmax(S, axis=0)] - 1000) <= FS / 512)


def test_spectrogram_short_signal_is_padded():
    t, f, S = sp.compute_spectrogram(np.ones(100), FS, window_size=256, hop=128)
    assert S.shape == (129, 1) and len(t) == 1


def test_spectrogram_bad_arguments():
    with pytest.raises(ValueError):
        sp.compute_spectrogram(tone(10), FS, window_size=1)
    with pytest.raises(ValueError):
        sp.compute_spectrogram(tone(10), FS, hop=0)
    with pytest.raises(ValueError):
        sp.compute_spectrogram(np.array([]), FS)


def test_spectrogram_real_speech_shape(speech):
    x, fs = speech
    t, f, S = sp.compute_spectrogram(x, fs)
    assert S.shape == (513, len(t)) and np.all(np.isfinite(S))
    assert t[-1] <= len(x) / fs


# ----------------------------------------------------------------------------
# filter_audio (LPF / HPF / BPF / Notch)
# ----------------------------------------------------------------------------
def band_energy(x, fs, lo, hi):
    f, m, _ = sp.compute_fft(x, fs)
    sel = (f >= lo) & (f <= hi)
    return np.sum(m[sel] ** 2) / np.sum(m ** 2)


def test_lowpass_highpass_bandpass_notch_on_tones():
    x = 0.3 * (tone(200) + tone(1000) + tone(3000))
    assert np.allclose(sp.filter_audio(x, FS, "lowpass", 500), 0.3 * tone(200), atol=1e-6)
    assert np.allclose(sp.filter_audio(x, FS, "highpass", 2000), 0.3 * tone(3000), atol=1e-6)
    assert np.allclose(sp.filter_audio(x, FS, "bandpass", (800, 1200)), 0.3 * tone(1000), atol=1e-6)
    assert np.allclose(sp.filter_audio(x, FS, "notch", 1000, width=20),
                       0.3 * (tone(200) + tone(3000)), atol=1e-6)


def test_filters_on_real_speech(speech):
    x, fs = speech
    lp = sp.filter_audio(x, fs, "lowpass", 1000)
    hp = sp.filter_audio(x, fs, "highpass", 1000)
    bp = sp.filter_audio(x, fs, "bandpass", (300, 3400))
    assert lp.shape == hp.shape == bp.shape == x.shape
    assert band_energy(lp, fs, 1200, fs / 2) < 1e-4
    assert band_energy(hp, fs, 0, 800) < 1e-4
    assert band_energy(bp, fs, 0, 250) < 1e-3 and band_energy(bp, fs, 3600, fs / 2) < 1e-3
    assert np.allclose(lp + hp, x, atol=1e-6)         # ideal LP + HP = original


def test_butterworth_is_smooth_and_attenuates(speech):
    x, fs = speech
    y = sp.filter_audio(x, fs, "lowpass", 2000, order=4)
    assert band_energy(y, fs, 6000, fs / 2) < 1e-3
    assert np.max(np.abs(y)) <= 1.0


def test_filter_audio_bad_arguments():
    x = tone(100)
    with pytest.raises(ValueError):
        sp.filter_audio(x, FS, "lowpass", FS / 2)           # at Nyquist: does nothing
    with pytest.raises(ValueError):
        sp.filter_audio(x, FS, "bandpass", (100, FS))       # high edge above Nyquist
    with pytest.raises(ValueError):
        sp.filter_audio(x, FS, "lowpass", -5)
    with pytest.raises(ValueError):
        sp.filter_audio(x, FS, "bandpass", 300)             # needs a (low, high) pair
    with pytest.raises(ValueError):
        sp.filter_audio(x, FS, "banana", 100)
    with pytest.raises(ValueError):
        sp.filter_audio(np.array([]), FS, "lowpass", 100)


# ----------------------------------------------------------------------------
# add_echo
# ----------------------------------------------------------------------------
def test_echo_matches_numpy_convolve():
    x = np.random.default_rng(2).uniform(-0.2, 0.2, 3000)
    d = int(round(0.05 * FS))
    h = np.zeros(d + 1)
    h[0], h[d] = 1, 0.5
    y = sp.add_echo(x, FS, 0.05, 0.5)
    assert len(y) == len(x) + d
    assert np.allclose(y, np.convolve(x, h), atol=1e-9)


def test_echo_impulse_response_position():
    x = np.zeros(2000)
    x[0] = 0.8
    y = sp.add_echo(x, FS, 0.1, 0.5)                         # d = 800 samples
    assert y[0] == pytest.approx(0.8) and y[800] == pytest.approx(0.4)
    assert np.count_nonzero(np.abs(y) > 1e-9) == 2


def test_echo_multiple_repeats():
    x = np.zeros(100)
    x[0] = 1.0
    y = sp.add_echo(x, 1000, 0.01, 0.5, n_echoes=3)          # d = 10
    assert np.allclose([y[0], y[10], y[20], y[30]], [1, .5, .25, .125])


def test_echo_short_kernel_uses_direct_convolution():
    x = np.random.default_rng(3).uniform(-0.3, 0.3, 200)
    y = sp.add_echo(x, 1000, 0.01, 0.6)                      # 11 taps -> convolve1d branch
    h = np.zeros(11)
    h[0], h[10] = 1, 0.6
    assert np.allclose(y, np.convolve(x, h), atol=1e-12)


def test_echo_never_exceeds_one():
    y = sp.add_echo(np.ones(500) * 0.99, FS, 0.01, 0.9)
    assert np.max(np.abs(y)) <= 1.0 + 1e-12


def test_echo_on_real_speech(speech):
    x, fs = speech
    y = sp.add_echo(x, fs, 0.25, 0.4)
    d = int(round(0.25 * fs))
    assert len(y) == len(x) + d and np.max(np.abs(y)) <= 1.0
    assert np.allclose(y[:d], x[:d], atol=1e-9)              # before the echo starts


@pytest.mark.parametrize("delay,decay", [(0, 0.5), (-1, 0.5), (0.1, -0.1), (0.1, 1.0),
                                         (1e-9, 0.5)])
def test_echo_bad_arguments(delay, decay):
    with pytest.raises(ValueError):
        sp.add_echo(tone(100), FS, delay, decay)


def test_echo_bad_signal_and_count():
    with pytest.raises(ValueError):
        sp.add_echo(np.array([]), FS, 0.1, 0.5)
    with pytest.raises(ValueError):
        sp.add_echo(tone(100), FS, 0.1, 0.5, n_echoes=0)
    with pytest.raises(ValueError):
        sp.add_echo(tone(100), FS, 5.0, 0.5, n_echoes=50)   # absurdly long


# ----------------------------------------------------------------------------
# reduce_noise
# ----------------------------------------------------------------------------
def make_noisy(seed=0, noise_std=0.2):
    """1 s of silence-with-noise, then 2 s of tones+noise."""
    rng = np.random.default_rng(seed)
    n = 3 * FS
    clean = np.zeros(n)
    t = np.arange(FS, n) / FS
    clean[FS:] = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 1200 * t)
    noise = noise_std * rng.standard_normal(n)
    return clean, clean + noise, noise[:FS]


def test_noise_reduction_improves_snr_auto():
    clean, noisy, _ = make_noisy()
    out = sp.reduce_noise(noisy, FS)
    assert len(out) == len(noisy)
    assert snr_db(clean, out) > snr_db(clean, noisy) + 3


def test_noise_reduction_with_noise_duration():
    clean, noisy, _ = make_noisy(1)
    out = sp.reduce_noise(noisy, FS, noise_duration=0.9)
    assert snr_db(clean, out) > snr_db(clean, noisy) + 3


def test_noise_reduction_with_noise_clip():
    clean, noisy, clip = make_noisy(2)
    out = sp.reduce_noise(noisy, FS, noise_clip=clip)
    assert snr_db(clean, out) > snr_db(clean, noisy) + 3


def test_noise_reduction_stronger_alpha_removes_more():
    clean, noisy, _ = make_noisy(3)
    weak = sp.reduce_noise(noisy, FS, alpha=0.5)
    strong = sp.reduce_noise(noisy, FS, alpha=3.0)
    assert np.mean(strong[:FS] ** 2) < np.mean(weak[:FS] ** 2)       # quieter in the noise-only part


def test_noise_reduction_perfect_reconstruction_when_nothing_removed():
    """alpha=0 means 'subtract nothing': STFT -> overlap-add must give x back exactly."""
    x = np.random.default_rng(4).standard_normal(5000)
    out = sp.reduce_noise(x, FS, alpha=0.0)
    assert np.allclose(out, x, atol=1e-8)


def test_noise_reduction_keeps_a_clean_signal_with_a_pause():
    """No noise at all (silent pause + tones): the signal must survive almost untouched."""
    x = np.zeros(3 * FS)
    x[FS:] = 0.5 * tone(440, n=2 * FS) + 0.3 * tone(1200, n=2 * FS)
    out = sp.reduce_noise(x, FS)
    assert np.corrcoef(x, out)[0, 1] > 0.99


def test_noise_reduction_on_real_speech(speech):
    x, fs = speech
    noisy = x + 0.03 * np.random.default_rng(5).standard_normal(len(x))
    out = sp.reduce_noise(noisy, fs)
    assert out.shape == x.shape and np.all(np.isfinite(out))
    assert snr_db(x, out) > snr_db(x, noisy) + 3


def test_noise_reduction_bad_arguments():
    x = tone(100, n=4000)
    with pytest.raises(ValueError):
        sp.reduce_noise(x, FS, method="magic")
    with pytest.raises(ValueError):
        sp.reduce_noise(x, FS, alpha=-1)
    with pytest.raises(ValueError):
        sp.reduce_noise(x, FS, beta=2)
    with pytest.raises(ValueError):
        sp.reduce_noise(x, FS, hop=2000)                     # hop > window
    with pytest.raises(ValueError):
        sp.reduce_noise(x, FS, noise_duration=0)
    with pytest.raises(ValueError):
        sp.reduce_noise(x, FS, noise_duration=100)           # longer than the signal
    with pytest.raises(ValueError):
        sp.reduce_noise(x, FS, noise_clip=np.zeros(10))      # shorter than one window
    with pytest.raises(ValueError):
        sp.reduce_noise(np.zeros(100), FS)                   # shorter than one window
    with pytest.raises(ValueError):
        sp.reduce_noise(np.array([]), FS)


# ----------------------------------------------------------------------------
# Whole pipeline on real audio (what the GUI will do)
# ----------------------------------------------------------------------------
def test_full_pipeline_real_audio(speech, tmp_path):
    x, fs = speech
    y = sp.filter_audio(x, fs, "bandpass", (300, 3400))
    y = sp.reduce_noise(y, fs)
    y = sp.add_echo(y, fs, 0.2, 0.3)
    out = tmp_path / "result.wav"
    sp.save_audio(out, y, fs)
    z, fs2 = sp.load_audio(out)
    assert fs2 == fs and len(z) == len(y)
    assert np.allclose(z, np.clip(y, -1, 1), atol=1 / 32768)
