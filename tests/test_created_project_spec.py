"""Tests for the items the project brief asks of Student 1 (DTFT) and Student 2 (convolution, difference equation)."""
import numpy as np
import pytest
from scipy import signal as sps

from created_signal import processing as P
from created_signal import reference as R

rng = np.random.default_rng(5)


# ----------------------------------------------------------------- DTFT (loops)
def test_dtft_matches_fft_on_dft_grid():
    x = rng.standard_normal(64)
    w = 2 * np.pi * np.arange(64) / 64
    assert np.allclose(P.dtft(x, w), np.fft.fft(x), atol=1e-10)


def test_dtft_of_impulse_is_flat_and_delayed_impulse_has_linear_phase():
    w = np.linspace(-np.pi, np.pi, 33)
    assert np.allclose(P.dtft([1, 0, 0, 0], w), 1.0)
    X = P.dtft([0, 0, 1], w)                         # delta[n-2] -> exp(-j*2*w)
    assert np.allclose(X, np.exp(-2j * w))


def test_dtft_of_cosine_peaks_at_its_angular_frequency():
    w0 = 0.8
    x = np.cos(w0 * np.arange(200))
    w, X, mag, ph = P.dtft_spectrum(x, 200, 1001)
    peak_w = w[np.argmax(mag)]
    assert abs(abs(peak_w) - w0) < 0.01
    assert np.isclose(mag.max(), 100, rtol=0.02)     # N/2 for a unit-amplitude cosine


def test_dtft_real_signal_has_even_magnitude_odd_phase():
    x = rng.standard_normal(40)
    w, X, mag, ph = P.dtft_spectrum(x, 40, 101)       # odd count -> symmetric grid
    assert np.allclose(mag, mag[::-1], atol=1e-9)
    assert np.allclose(X, np.conj(X[::-1]), atol=1e-9)


def test_dtft_two_sines_plus_noise_shows_both_peaks():
    fs = 1000
    t = np.arange(512) / fs
    x = np.sin(2 * np.pi * 100 * t) + 0.5 * np.sin(2 * np.pi * 250 * t)
    x = P.add_noise(x, 20, seed=2)
    w, X, mag, ph = P.dtft_spectrum(x, 512, 512)
    pos = w > 0
    hz = w[pos] * fs / (2 * np.pi)
    top = hz[np.argsort(mag[pos])[-6:]]
    assert any(abs(top - 100) < 5) and any(abs(top - 250) < 5)


def test_dtft_validation():
    with pytest.raises(ValueError):
        P.dtft_spectrum(np.ones(10), 1, 10)
    with pytest.raises(ValueError):
        P.dtft(np.ones(2000), np.zeros(2000))         # 4M terms: refused


# ---------------------------------------------------- convolution vs frequency
def test_zero_padded_frequency_multiplication_equals_linear_convolution():
    x, h = rng.standard_normal(100), P.make_kernel("moving_average", 7)
    r = P.convolution_vs_frequency(x, h)
    assert len(r["direct"]) == 106
    assert np.allclose(r["direct"], np.convolve(x, h), atol=1e-12)
    assert np.allclose(r["padded"], r["direct"], atol=1e-10)


def test_without_zero_padding_it_is_circular_and_wrong_only_at_the_start():
    x, h = rng.standard_normal(100), P.make_kernel("moving_average", 7)
    r = P.convolution_vs_frequency(x, h)
    err = np.abs(r["err_circular"])
    assert err[:6].max() > 1e-3                      # the tail wrapped onto the first M-1 samples
    assert np.allclose(err[6:], 0, atol=1e-10)       # everything else is already correct


# ------------------------------------------------------ difference equation
def test_difference_equation_matches_scipy_lfilter():
    x = rng.standard_normal(300)
    for a, b in [(0.9, 0.1), (-0.5, 1.0), (0.0, 2.0), (0.99, 0.01)]:
        assert np.allclose(P.first_order_difference_equation(x, a, b),
                           sps.lfilter([b], [1, -a], x), atol=1e-12)


def test_difference_equation_impulse_response_is_b_times_a_to_the_n():
    delta = np.zeros(20)
    delta[0] = 1
    h = P.first_order_difference_equation(delta, 0.8, 0.5)
    assert np.allclose(h, 0.5 * 0.8 ** np.arange(20))


def test_difference_equation_step_settles_at_dc_gain():
    y = P.first_order_difference_equation(np.ones(500), 0.9, 0.1)
    assert y[-1] == pytest.approx(0.1 / (1 - 0.9), abs=1e-6)         # b / (1 - a) = 1


def test_difference_equation_stability():
    assert P.is_stable(0.9) and P.is_stable(-0.99)
    assert not P.is_stable(1.0) and not P.is_stable(-1.2)
    y = P.first_order_difference_equation(np.ones(100), 1.1, 1.0)
    assert abs(y[-1]) > 1e3                           # blows up


def test_difference_equation_through_filter_spec_and_reference():
    x = rng.standard_normal(200)
    spec = {"type": "first_order", "a": 0.7, "b": 0.3}
    assert np.allclose(P.apply_filter_spec(x, 1000, spec), R.ref_apply_filter_spec(x, 1000, spec))
    assert "first_order" in P.FILTER_KINDS


def test_moving_average_is_linear_convolution():
    x = rng.standard_normal(50)
    y = P.apply_conv_filter(x, "moving_average", 5)
    ref = np.convolve(x, np.ones(5) / 5, mode="same")
    assert np.allclose(y, ref)
