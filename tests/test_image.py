
import numpy as np
import pytest

from PIL import Image
from scipy.signal import convolve2d as scipy_convolve2d

from image_signal import (
    load_image,
    spatial_filter,
    image_fft,
    frequency_filter
)


def ref_convolution(img, kernel):
    return scipy_convolve2d(
        img,
        kernel,
        mode="same",
        boundary="symm"
    )


def test_load_image(tmp_path):
    rgb = np.zeros((6, 8, 3), dtype=np.uint8)
    rgb[..., 0] = 255

    path = tmp_path / "rgb.png"
    Image.fromarray(rgb).save(path)

    result = load_image(path)

    assert result.shape == (6, 8)
    assert result.dtype == np.float64
    assert result.min() >= 0
    assert result.max() <= 255


def test_load_image_invalid_path():
    with pytest.raises(ValueError):
        load_image("missing_image.png")


@pytest.mark.parametrize(
    "name",
    [
        "box",
        "gaussian",
        "sharpen",
        "sobel_x",
        "sobel_y",
        "sobel",
        "prewitt",
        "laplacian"
    ]
)
def test_spatial_filter(name):
    img = np.random.default_rng(10).uniform(
        0, 255, (19, 23)
    )

    result = spatial_filter(img, name)

    assert result.shape == img.shape
    assert result.dtype == np.float64
    assert np.isfinite(result).all()
    assert result.min() >= 0
    assert result.max() <= 255


def test_box_blur_comparison():
    img = np.random.default_rng(11).uniform(
        0, 255, (15, 17)
    )

    kernel = np.ones((3, 3)) / 9

    expected = ref_convolution(img, kernel)
    actual = spatial_filter(img, "box", size=3)

    assert np.allclose(
        actual[1:-1, 1:-1],
        expected[1:-1, 1:-1],
        atol=1e-6
    )


def test_gaussian_blur_comparison():
    img = np.random.default_rng(12).uniform(
        0, 255, (15, 17)
    )

    size = 5
    sigma = 1.2

    axis = np.arange(size) - size // 2
    xx, yy = np.meshgrid(axis, axis)

    kernel = np.exp(
        -(xx**2 + yy**2) / (2 * sigma**2)
    )

    kernel /= kernel.sum()

    expected = ref_convolution(img, kernel)
    actual = spatial_filter(
        img,
        "gaussian",
        size=size,
        sigma=sigma
    )

    assert np.allclose(
        actual[2:-2, 2:-2],
        np.clip(expected, 0, 255)[2:-2, 2:-2],
        atol=1e-6
    )


def test_sharpen_comparison():
    img = np.random.default_rng(13).uniform(
        0, 255, (15, 17)
    )

    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=float)

    expected = np.clip(
        ref_convolution(img, kernel),
        0,
        255
    )

    actual = spatial_filter(img, "sharpen")

    assert np.allclose(
        actual[1:-1, 1:-1],
        expected[1:-1, 1:-1],
        atol=1e-6
    )


def test_image_fft_comparison():
    img = np.random.default_rng(14).uniform(
        0, 255, (9, 12)
    )

    spectrum, log_mag, angle = image_fft(img)

    reference = np.fft.fftshift(
        np.fft.fft2(img)
    )

    assert np.allclose(spectrum, reference, atol=1e-6)

    assert np.allclose(
        log_mag,
        np.log1p(np.abs(reference)),
        atol=1e-6
    )

    assert np.allclose(
        angle,
        np.angle(reference),
        atol=1e-6
    )


@pytest.mark.parametrize(
    "filter_type",
    ["lowpass", "highpass"]
)
@pytest.mark.parametrize(
    "kind",
    ["ideal", "gaussian", "butterworth"]
)
def test_frequency_filter_comparison(filter_type, kind):
    img = np.random.default_rng(15).uniform(
        0, 255, (16, 18)
    )

    cutoff = 5.0
    order = 2

    rows, cols = img.shape

    u = np.arange(rows) - rows // 2
    v = np.arange(cols) - cols // 2

    distance = np.sqrt(
        u[:, None]**2 + v[None, :]**2
    )

    if kind == "ideal":
        low = (distance <= cutoff).astype(float)

    elif kind == "gaussian":
        low = np.exp(
            -(distance**2) / (2 * cutoff**2)
        )

    else:
        low = 1 / (
            1 + (distance / cutoff)**(2 * order)
        )

    mask = (
        low if filter_type == "lowpass"
        else 1 - low
    )

    F = np.fft.fftshift(np.fft.fft2(img))
    G = F * mask

    expected = np.fft.ifft2(
        np.fft.ifftshift(G)
    ).real

    expected = np.clip(expected, 0, 255)

    out, log_mag, angle = frequency_filter(
        img,
        filter_type,
        cutoff,
        kind,
        order
    )

    assert np.allclose(out, expected, atol=1e-6)

    assert np.allclose(
        log_mag,
        np.log1p(np.abs(G)),
        atol=1e-6
    )

    valid = np.abs(G) > 1e-10

    phase_error = np.angle(
        np.exp(
            1j * (angle[valid] - np.angle(G)[valid])
        )
    )

    assert np.allclose(
        phase_error,
        0.0,
        atol=1e-6
    )


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((0, 3)),
        np.zeros((3, 3, 3)),
        np.array([[np.nan]])
    ]
)
def test_invalid_images(bad):
    with pytest.raises(ValueError):
        image_fft(bad)


def test_invalid_filter_parameters():
    with pytest.raises(ValueError):
        spatial_filter(
            np.ones((5, 5)),
            "box",
            size=2
        )

    with pytest.raises(ValueError):
        frequency_filter(
            np.ones((5, 5)),
            "lowpass",
            0
        )


@pytest.mark.parametrize(
    "name",
    [
        "box",
        "gaussian",
        "sharpen",
        "sobel_x",
        "sobel_y",
        "sobel",
        "prewitt",
        "laplacian"
    ]
)
def test_spatial_filter(name):
    img = np.random.default_rng(10).uniform(
        0, 255, (19, 23)
    )

    result = spatial_filter(img, name)

    assert result.shape == img.shape
    assert result.dtype == np.float64
    assert np.isfinite(result).all()

    if name in ("sobel_x", "sobel_y"):
        # Signed gradients can contain negative values
        assert np.any(result < 0)
        assert np.any(result > 0)

    else:
        # Other filters return display-range images
        assert result.min() >= 0
        assert result.max() <= 255



def test_sobel_gradients_keep_signed_values():
    # Horizontal gradient
    img = np.tile(
        np.arange(9, dtype=np.float64),
        (9, 1)
    )

    gx = spatial_filter(img, "sobel_x")
    gy = spatial_filter(img, "sobel_y")

    # Sobel X should preserve negative values
    assert np.any(gx < 0)

    # Sobel Y should be approximately zero
    # because there is no vertical intensity change
    assert np.allclose(gy, 0.0, atol=1e-10)