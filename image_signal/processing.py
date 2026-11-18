
"""Image-processing core API for the GUI.

Images are 2D grayscale float64 arrays. Input pixel values are expected in
[0, 255]. Functions return arrays/data only; this module has no GUI dependency.
"""

from pathlib import Path
import numpy as np
from PIL import Image

from core.convolution import convolve2d
from core.filters import freq_filter_2d
from core.fourier import fft2
from core.utils import fftshift, magnitude, phase


def _validate_image(img):
    arr = np.asarray(img, dtype=np.float64)

    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("img must be a non-empty 2D grayscale image")

    if not np.isfinite(arr).all():
        raise ValueError("img must contain only finite values")

    return arr


def _odd_size(size):
    if (
        isinstance(size, bool)
        or not isinstance(size, (int, np.integer))
        or size < 1
        or size % 2 == 0
    ):
        raise ValueError("size must be a positive odd integer")

    return int(size)


def load_image(path: str) -> np.ndarray:
    """Load an image as grayscale float64 pixels in [0, 255]."""

    if not isinstance(path, (str, Path)) or not str(path):
        raise ValueError("path must be a non-empty string")

    try:
        with Image.open(path) as source:
            return np.asarray(
                source.convert("L"),
                dtype=np.float64
            ).copy()

    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load image: {path}") from exc


def spatial_filter(
    img,
    name: str,
    size: int = 3,
    sigma: float = 1.0
) -> np.ndarray:
    """Apply spatial filters and return a same-size float64 image.

    Supported names:
    box, gaussian, sharpen, sobel_x, sobel_y,
    sobel, prewitt, laplacian
    """

    image = _validate_image(img)
    size = _odd_size(size)
    name = str(name).lower()

    if name == "box":

        kernel = np.ones(
            (size, size),
            dtype=np.float64
        ) / (size * size)

        result = convolve2d(
            image,
            kernel,
            padding="reflect"
        )

    elif name == "gaussian":

        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be positive")

        axis = np.arange(size, dtype=np.float64) - size // 2
        xx, yy = np.meshgrid(axis, axis)

        kernel = np.exp(
            -(xx * xx + yy * yy) / (2.0 * sigma * sigma)
        )

        kernel /= kernel.sum()

        result = convolve2d(
            image,
            kernel,
            padding="reflect"
        )

    else:

        kernels = {
            "sharpen": np.array([
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ], dtype=float),

            "sobel_x": np.array([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ], dtype=float),

            "sobel_y": np.array([
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ], dtype=float),

            "prewitt_x": np.array([
                [-1, 0, 1],
                [-1, 0, 1],
                [-1, 0, 1]
            ], dtype=float),

            "prewitt_y": np.array([
                [-1, -1, -1],
                [0, 0, 0],
                [1, 1, 1]
            ], dtype=float),

            "laplacian": np.array([
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]
            ], dtype=float),
        }

        if name == "sobel":

            gx = convolve2d(
                image,
                kernels["sobel_x"],
                padding="reflect"
            )

            gy = convolve2d(
                image,
                kernels["sobel_y"],
                padding="reflect"
            )

            result = np.hypot(gx, gy)

        elif name == "prewitt":

            gx = convolve2d(
                image,
                kernels["prewitt_x"],
                padding="reflect"
            )

            gy = convolve2d(
                image,
                kernels["prewitt_y"],
                padding="reflect"
            )

            result = np.hypot(gx, gy)

        elif name in kernels:

            result = convolve2d(
                image,
                kernels[name],
                padding="reflect"
            )

            if name == "laplacian":
                result = np.abs(result)

        else:
            raise ValueError(
                "Unsupported filter name: " + name
            )

    if name in ("sobel_x", "sobel_y"):
        return result.astype(np.float64, copy=False)

    return np.clip(
        result,
        0.0,
        255.0
    ).astype(np.float64, copy=False)


def image_fft(img):
    """
    Returns:
    (F_shifted, log_magnitude, phase)
    """

    image = _validate_image(img)

    spectrum = fftshift(fft2(image))

    return (
        spectrum,
        np.log1p(magnitude(spectrum)),
        phase(spectrum)
    )


def frequency_filter(
    img,
    filter_type,
    cutoff,
    kind="ideal",
    order=2
):
    """
    Returns:
    (output_img, log_magnitude_of_output, phase_of_output)
    """

    image = _validate_image(img)

    output, _input_spectrum, output_spectrum = freq_filter_2d(
        image,
        filter_type,
        cutoff,
        kind=kind,
        order=order
    )

    output = np.clip(
        output,
        0.0,
        255.0
    ).astype(np.float64, copy=False)

    return (
        output,
        np.log1p(magnitude(output_spectrum)),
        phase(output_spectrum)
    )