import numpy as np
import pytest
from core import fourier

rng = np.random.default_rng(0)
LENGTHS = [1, 2, 3, 8, 15, 64, 100, 97, 125, 256, 1000, 2310]


@pytest.mark.parametrize("n", LENGTHS)
def test_dft_matches_numpy(n):
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    assert np.allclose(fourier.dft(x), np.fft.fft(x), atol=1e-8)


@pytest.mark.parametrize("n", LENGTHS)
def test_fft_matches_numpy(n):
    x = rng.standard_normal(n)
    assert np.allclose(fourier.fft(x), np.fft.fft(x), atol=1e-8)


@pytest.mark.parametrize("n", LENGTHS)
def test_ifft_and_idft_round_trip(n):
    x = rng.standard_normal(n)
    assert np.allclose(fourier.ifft(fourier.fft(x)).real, x, atol=1e-8)
    assert np.allclose(fourier.idft(fourier.dft(x)).real, x, atol=1e-8)
    assert np.allclose(fourier.ifft(np.fft.fft(x)), x, atol=1e-8)


def test_fft_zero_padding():
    x = rng.standard_normal(10)
    assert np.allclose(fourier.fft(x, n=16), np.fft.fft(x, n=16), atol=1e-8)


def test_fft_peak_at_right_frequency():
    fs, f0 = 1000, 50
    t = np.arange(1000) / fs
    X = fourier.fft(np.sin(2 * np.pi * f0 * t))
    assert np.argmax(np.abs(X[:500])) == f0


def test_parseval():
    x = rng.standard_normal(128)
    X = fourier.fft(x)
    assert np.isclose(np.sum(x ** 2), np.sum(np.abs(X) ** 2) / len(x))


@pytest.mark.parametrize("shape", [(16, 16), (30, 40), (7, 9), (64, 32)])
def test_fft2_matches_numpy(shape):
    img = rng.random(shape)
    assert np.allclose(fourier.fft2(img), np.fft.fft2(img), atol=1e-8)


@pytest.mark.parametrize("shape", [(16, 16), (30, 40), (7, 9)])
def test_ifft2_round_trip(shape):
    img = rng.random(shape)
    assert np.allclose(fourier.ifft2(fourier.fft2(img)).real, img, atol=1e-8)


def test_bad_input_raises():
    with pytest.raises(ValueError):
        fourier.fft(np.zeros((3, 3)))
    with pytest.raises(ValueError):
        fourier.fft2(np.zeros(5))
    with pytest.raises(ValueError):
        fourier.fft(np.array([]))


@pytest.mark.parametrize("n", [10007, 44100, 132300])
def test_fft_large_awkward_lengths(n):
    """Audio-sized and prime lengths must be fast and correct (Bluestein path)."""
    x = rng.standard_normal(n)
    X = fourier.fft(x)
    assert np.allclose(X, np.fft.fft(x), atol=1e-6)
    assert np.allclose(fourier.ifft(X).real, x, atol=1e-8)


@pytest.mark.parametrize("n", [55125, 220500])
def test_fft_large_lengths_with_odd_factor(n):
    """Regression: Bluestein used int32 indices on Windows, so n*n overflowed for n > 46340."""
    x = np.random.default_rng(7).standard_normal(n)
    assert np.allclose(fourier.fft(x), np.fft.fft(x), atol=1e-6)
