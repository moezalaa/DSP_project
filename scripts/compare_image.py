
import numpy as np
from scipy.signal import convolve2d as scipy_convolve2d
from pathlib import Path
from PIL import Image

from image_signal import (
    load_image,
    spatial_filter,
    image_fft,
    frequency_filter
)
from scripts.demo_image import IMAGE_PATH


# ----------------------------------
# Comparison metrics
# ----------------------------------

def calculate_metrics(ours, reference):
    ours = np.asarray(ours)
    reference = np.asarray(reference)

    difference = np.abs(ours - reference)

    mae = np.mean(difference)
    rmse = np.sqrt(np.mean(difference ** 2))
    max_error = np.max(difference)

    if rmse == 0:
        psnr = float("inf")
    else:
        psnr = 20 * np.log10(255.0 / rmse)

    return mae, rmse, max_error, psnr


def compare(name, ours, reference):
    mae, rmse, max_error, psnr = calculate_metrics(
        ours, reference
    )

    print(f"\n{name}")
    print("-" * 55)
    print(f"MAE       : {mae:.10f}")
    print(f"RMSE      : {rmse:.10f}")
    print(f"Max Error : {max_error:.10f}")
    print(f"PSNR      : {psnr:.4f} dB")

    return mae, rmse, max_error, psnr


# ----------------------------------
# Reference spatial convolution
# ----------------------------------

def reference_convolution(img, kernel):
    return scipy_convolve2d(
        img,
        kernel,
        mode="same",
        boundary="symm"
    )


def reference_spatial_filter(img, name, size=3, sigma=1.0):

    if name == "box":
        kernel = np.ones((size, size)) / (size * size)

    elif name == "gaussian":
        axis = np.arange(size) - size // 2
        xx, yy = np.meshgrid(axis, axis)

        kernel = np.exp(
            -(xx**2 + yy**2) / (2 * sigma**2)
        )
        kernel /= kernel.sum()

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
            ], dtype=float)
        }

        if name == "sobel":
            gx = reference_convolution(
                img, kernels["sobel_x"]
            )
            gy = reference_convolution(
                img, kernels["sobel_y"]
            )
            return np.clip(np.hypot(gx, gy), 0, 255)

        if name == "prewitt":
            gx = reference_convolution(
                img, kernels["prewitt_x"]
            )
            gy = reference_convolution(
                img, kernels["prewitt_y"]
            )
            return np.clip(np.hypot(gx, gy), 0, 255)

        kernel = kernels[name]

    result = reference_convolution(img, kernel)

    if name == "laplacian":
        result = np.abs(result)

    return np.clip(result, 0, 255)


# ----------------------------------
# Frequency reference using NumPy
# ----------------------------------

def reference_frequency_filter(
    img, filter_type, cutoff, kind="ideal", order=2
):
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

    elif kind == "butterworth":
        low = 1 / (
            1 + (distance / cutoff)**(2 * order)
        )

    else:
        raise ValueError("Unknown filter kind")

    mask = (
        low if filter_type == "lowpass"
        else 1 - low
    )

    F = np.fft.fftshift(np.fft.fft2(img))
    G = F * mask

    output = np.fft.ifft2(
        np.fft.ifftshift(G)
    ).real

    return (
        np.clip(output, 0, 255),
        G
    )


# ----------------------------------
# Run all comparisons
# ----------------------------------

def main():

    # Load the real test image
    IMAGE_PATH = Path(__file__).with_name("test_image.jpg")

    img = load_image(IMAGE_PATH)

    # Resize to 128x128 to make manual FFT comparison faster
    img = np.asarray(
        Image.fromarray(
            np.clip(img, 0, 255).astype(np.uint8)
        ).resize((128, 128)),
        dtype=np.float64
    )

    print("=" * 55)
    print("IMAGE PROCESSING COMPARISON")
    print("=" * 55)

    # Spatial filters
    spatial_names = [
        "box",
        "gaussian",
        "sharpen",
        "sobel_x",
        "sobel_y",
        "sobel",
        "prewitt",
        "laplacian"
    ]

    for name in spatial_names:

        ours = spatial_filter(
            img,
            name,
            size=5 if name in ("box", "gaussian") else 3,
            sigma=1.2
        )

        reference = reference_spatial_filter(
            img,
            name,
            size=5 if name in ("box", "gaussian") else 3,
            sigma=1.2
        )

        # Ignore border pixels because padding conventions
        # can differ between implementations.
        if name in ("box", "gaussian"):
            margin = 2
        else:
            margin = 1

        compare(
            f"Spatial: {name}",
            ours[margin:-margin, margin:-margin],
            reference[margin:-margin, margin:-margin]
        )

    # FFT comparison
    ours_F, ours_log, ours_phase = image_fft(img)

    ref_F = np.fft.fftshift(
        np.fft.fft2(img)
    )

    compare("FFT Complex Spectrum", ours_F, ref_F)

    compare(
        "FFT Log Magnitude",
        ours_log,
        np.log1p(np.abs(ref_F))
    )

    # Phase is compared circularly because angles wrap at +/- pi.
    phase_difference = np.angle(
        np.exp(1j * (ours_phase - np.angle(ref_F)))
    )

    compare(
        "FFT Phase",
        phase_difference,
        np.zeros_like(phase_difference)
    )

    # Frequency filters
    for filter_type in ("lowpass", "highpass"):

        for kind in ("ideal", "gaussian", "butterworth"):

            ours_img, ours_log, ours_phase = frequency_filter(
                img,
                filter_type,
                cutoff=20,
                kind=kind,
                order=2
            )

            ref_img, ref_G = reference_frequency_filter(
                img,
                filter_type,
                cutoff=20,
                kind=kind,
                order=2
            )

            compare(
                f"Frequency Image: {filter_type} {kind}",
                ours_img,
                ref_img
            )

            compare(
                f"Frequency Log Magnitude: {filter_type} {kind}",
                ours_log,
                np.log1p(np.abs(ref_G))
            )

            phase_difference = np.angle(
                np.exp(1j * (ours_phase - np.angle(ref_G)))
            )

            valid = np.abs(ref_G) > 1e-10

            if np.any(valid):
                compare(
                    f"Frequency Phase: {filter_type} {kind}",
                    phase_difference[valid],
                    np.zeros(np.count_nonzero(valid))
                )

    print("\nComparison finished.")


if __name__ == "__main__":
    main()