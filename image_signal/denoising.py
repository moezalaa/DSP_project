from core.fourier import fft2, ifft2
from core.utils import fftshift, magnitude,ifftshift
import numpy as np


def add_periodic_noise(img, amplitude=30, fx=8, fy=6):
    """
    Add periodic sinusoidal noise to a grayscale image.

    amplitude: Noise strength.
    fx: Horizontal frequency in cycles across image width.
    fy: Vertical frequency in cycles across image height.

    Returns a noisy float64 image.
    """

    image = np.asarray(img, dtype=np.float64)

    if image.ndim != 2 or image.size == 0:
        raise ValueError("Image must be a non-empty grayscale image")

    if amplitude < 0:
        raise ValueError("Amplitude must be non-negative")

    rows, cols = image.shape

    x = np.arange(cols)
    y = np.arange(rows)

    xx, yy = np.meshgrid(x, y)

    noise = amplitude * np.sin(
        2 * np.pi * (
            fx * xx / cols +
            fy * yy / rows
        )
    )

    noisy_image = image + noise

    return noisy_image


def analyze_noisy_image(noisy_img):
    """
    Calculate the centered log magnitude spectrum
    of a noisy grayscale image.
    """

    spectrum = fftshift(fft2(noisy_img))

    log_magnitude = np.log1p(magnitude(spectrum))

    return spectrum, log_magnitude


def create_notch_mask(shape, fx=8, fy=6, radius=3):
    rows, cols = shape

    y, x = np.ogrid[:rows, :cols]

    center_y = rows // 2
    center_x = cols // 2

    mask = np.ones((rows, cols), dtype=np.float64)

    # Positive frequency peak
    d1 = np.sqrt(
        (y - (center_y + fy))**2 +
        (x - (center_x + fx))**2
    )

    # Negative frequency peak
    d2 = np.sqrt(
        (y - (center_y - fy))**2 +
        (x - (center_x - fx))**2
    )

    mask[(d1 <= radius) | (d2 <= radius)] = 0

    return mask


def remove_periodic_noise(noisy_img, fx=8, fy=6, radius=3):
    spectrum = fftshift(fft2(noisy_img))

    mask = create_notch_mask(
        noisy_img.shape,
        fx=fx,
        fy=fy,
        radius=radius
    )

    filtered_spectrum = spectrum * mask

    restored = ifft2(ifftshift(filtered_spectrum)).real

    return restored, mask