import numpy as np
import pytest
from core import filters, fourier

FS = 1000
T = np.arange(1000) / FS


def tone(f):
    return np.sin(2 * np.pi * f * T)


def energy(x):
    return np.mean(x ** 2)


def test_lowpass_keeps_low_removes_high():
    y, _, _ = filters.freq_filter_1d(tone(20) + tone(200), FS, "lowpass", 100)
    assert np.allclose(y, tone(20), atol=1e-6)


def test_highpass_keeps_high_removes_low():
    y, _, _ = filters.freq_filter_1d(tone(20) + tone(200), FS, "highpass", 100)
    assert np.allclose(y, tone(200), atol=1e-6)


def test_bandpass():
    x = tone(20) + tone(100) + tone(300)
    y, _, _ = filters.freq_filter_1d(x, FS, "bandpass", (80, 120))
    assert np.allclose(y, tone(100), atol=1e-6)


def test_notch_removes_one_frequency():
    y, _, _ = filters.freq_filter_1d(tone(50) + tone(120), FS, "notch", 50, width=4)
    assert np.allclose(y, tone(120), atol=1e-6)


def test_butterworth_is_smoother_and_attenuates():
    x = tone(20) + tone(300)
    y, _, _ = filters.freq_filter_1d(x, FS, "lowpass", 100, order=4)
    assert energy(y - tone(20)) < 0.001


def test_output_is_real_and_same_length():
    y, X, Y = filters.freq_filter_1d(np.random.rand(777), FS, "lowpass", 100)
    assert y.shape == (777,) and np.isrealobj(y)
    assert X.shape == Y.shape == (777,)


@pytest.mark.parametrize("args", [("bad", 10), ("lowpass", -5), ("bandpass", 10)])
def test_1d_bad_params_raise(args):
    with pytest.raises(ValueError):
        filters.freq_filter_1d(tone(10), FS, *args)


def test_bandpass_bad_range():
    with pytest.raises(ValueError):
        filters.freq_filter_1d(tone(10), FS, "bandpass", (200, 100))


@pytest.mark.parametrize("kind", ["ideal", "gaussian", "butterworth"])
def test_mask_2d_lowpass_highpass_are_complementary(kind):
    lp = filters.freq_mask_2d((64, 48), "lowpass", 10, kind)
    hp = filters.freq_mask_2d((64, 48), "highpass", 10, kind)
    assert np.allclose(lp + hp, 1.0)
    assert lp.shape == (64, 48)
    assert lp.min() >= 0 and lp.max() <= 1
    assert lp[32, 24] == pytest.approx(1.0)          # DC at the center


def test_mask_2d_ideal_radius():
    m = filters.freq_mask_2d((65, 65), "lowpass", 5, "ideal")
    assert m[32, 32 + 5] == 1 and m[32, 32 + 6] == 0


def test_freq_filter_2d_lowpass_blurs_and_highpass_removes_dc():
    rng = np.random.default_rng(2)
    img = rng.random((64, 64)) * 255
    low, _, _ = filters.freq_filter_2d(img, "lowpass", 8, "gaussian")
    high, _, _ = filters.freq_filter_2d(img, "highpass", 8, "gaussian")
    assert low.std() < img.std()
    assert abs(high.mean()) < 1.0
    assert low.shape == high.shape == img.shape and np.isrealobj(low)


def test_freq_filter_2d_full_pass_returns_same_image():
    img = np.random.default_rng(3).random((30, 20))
    out, _, _ = filters.freq_filter_2d(img, "lowpass", 10_000, "ideal")
    assert np.allclose(out, img, atol=1e-8)


def test_apply_filter_1d_contract():
    out, mag, ph = filters.apply_filter(tone(20) + tone(200), "lowpass", fs=FS, cutoff=100)
    assert out.shape == mag.shape == ph.shape == (1000,)
    assert np.argmax(mag[:500]) == 20
    assert ph.min() >= -np.pi and ph.max() <= np.pi


def test_apply_filter_2d_contract():
    img = np.random.default_rng(4).random((32, 40))
    out, mag, ph = filters.apply_filter(img, "highpass", cutoff=5, kind="butterworth", order=2)
    assert out.shape == mag.shape == ph.shape == (32, 40)
