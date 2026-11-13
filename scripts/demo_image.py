
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from image_signal import (
    load_image,
    spatial_filter,
    image_fft,
    frequency_filter
)

# Load your image
IMAGE_PATH = Path(__file__).with_name("test_image.jpg")
img = load_image(IMAGE_PATH)

# Apply spatial filters
results = {
    "Original": img,
    "Box Blur": spatial_filter(img, "box"),
    "Gaussian Blur": spatial_filter(
        img, "gaussian", size=5, sigma=1.2
    ),
    "Sharpen": spatial_filter(img, "sharpen"),
    "Sobel X": spatial_filter(img, "sobel_x"),
    "Sobel Y": spatial_filter(img, "sobel_y"),
    "Sobel": spatial_filter(img, "sobel"),
    "Prewitt": spatial_filter(img, "prewitt"),
    "Laplacian": spatial_filter(img, "laplacian"),
}

# Display spatial results
fig, axes = plt.subplots(3, 3, figsize=(13, 12))

for ax, (name, result) in zip(axes.flat, results.items()):

    if name in ("Sobel X", "Sobel Y"):
        limit = np.max(np.abs(result))
        limit = max(limit, 1e-10)

        ax.imshow(
            result,
            cmap="gray",
            vmin=-limit,
            vmax=limit
        )
    else:
        ax.imshow(
            result,
            cmap="gray",
            vmin=0,
            vmax=255
        )

    ax.set_title(name)
    ax.axis("off")

plt.tight_layout()
plt.show()

# FFT
F_shifted, log_magnitude, phase = image_fft(img)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img, cmap="gray")
axes[0].set_title("Original")

axes[1].imshow(log_magnitude, cmap="gray")
axes[1].set_title("FFT Log Magnitude")

axes[2].imshow(phase, cmap="gray")
axes[2].set_title("FFT Phase")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()

# Frequency filters
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, kind in zip(
    axes,
    ["ideal", "gaussian", "butterworth"]
):
    output, log_mag, output_phase = frequency_filter(
        img,
        filter_type="lowpass",
        cutoff=30,
        kind=kind,
        order=2
    )

    ax.imshow(output, cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"{kind} Low-pass")
    ax.axis("off")

plt.tight_layout()
plt.show()