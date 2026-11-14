import numpy as np
import pytest
from scipy import signal
from core import convolution

rng = np.random.default_rng(1)


@pytest.mark.parametrize("mode", ["full", "same", "valid"])
@pytest.mark.parametrize("N,M", [(10, 3), (10, 4), (5, 5), (20, 2), (7, 1), (3, 8)])
def test_convolve1d_matches_numpy(mode, N, M):
    if mode == "valid" and M > N:
        pytest.skip("valid with longer kernel: numpy swaps arguments")
    x, h = rng.standard_normal(N), rng.standard_normal(M)
    assert np.allclose(convolution.convolve1d(x, h, mode), np.convolve(x, h, mode))


def test_convolve1d_identity_and_shift():
    x = rng.standard_normal(8)
    assert np.allclose(convolution.convolve1d(x, [1.0], "full"), x)
    shifted = convolution.convolve1d(x, [0, 0, 1.0], "full")
    assert np.allclose(shifted[2:10], x)


def test_convolve1d_commutative():
    x, h = rng.standard_normal(9), rng.standard_normal(4)
    assert np.allclose(convolution.convolve1d(x, h), convolution.convolve1d(h, x))


@pytest.mark.parametrize("ksize", [(3, 3), (5, 5), (3, 5), (4, 4), (1, 1), (2, 3)])
def test_convolve2d_matches_scipy(ksize):
    img = rng.random((20, 30))
    k = rng.standard_normal(ksize)
    expected = signal.convolve2d(img, k, mode="same", boundary="fill")
    assert np.allclose(convolution.convolve2d(img, k), expected)


@pytest.mark.parametrize("padding,boundary", [("reflect", "symm"), ("zero", "fill")])
def test_convolve2d_padding_shapes(padding, boundary):
    img = rng.random((12, 15))
    k = np.ones((3, 3)) / 9
    out = convolution.convolve2d(img, k, padding=padding)
    assert out.shape == img.shape
    # interior pixels do not depend on padding
    ref = signal.convolve2d(img, k, mode="same", boundary="fill")
    assert np.allclose(out[1:-1, 1:-1], ref[1:-1, 1:-1])


def test_convolve2d_box_blur_keeps_constant_image():
    img = np.full((10, 10), 5.0)
    out = convolution.convolve2d(img, np.ones((3, 3)) / 9, padding="edge")
    assert np.allclose(out, 5.0)


def test_bad_input_raises():
    with pytest.raises(ValueError):
        convolution.convolve1d([1, 2, 3], [1], mode="wrong")
    with pytest.raises(ValueError):
        convolution.convolve1d([], [1])
    with pytest.raises(ValueError):
        convolution.convolve2d(np.zeros((3, 3)), np.ones((3, 3)), padding="wrong")
