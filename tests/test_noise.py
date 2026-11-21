
from image_signal.processing import load_image
from image_signal.denoising import add_periodic_noise, analyze_noisy_image, remove_periodic_noise
from scripts.compare_image import calculate_metrics

import matplotlib.pyplot as plt

img = load_image("scripts/test_image.jpg")

noisy = add_periodic_noise(
    img,
    amplitude=30,
    fx=8,
    fy=6
)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(noisy, cmap="gray")
plt.title("Periodic Noise")
plt.axis("off")

plt.tight_layout()
plt.show()


spectrum, log_magnitude = analyze_noisy_image(noisy)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(noisy, cmap="gray")
plt.title("Noisy Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(log_magnitude, cmap="gray")
plt.title("Noisy Image - FFT Spectrum")
plt.axis("off")

plt.tight_layout()
plt.show()


restored, mask = remove_periodic_noise(
    noisy,
    fx=8,
    fy=6,
    radius=3
)

plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(noisy, cmap="gray")
plt.title("Noisy")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(restored, cmap="gray")
plt.title("Restored")
plt.axis("off")

plt.tight_layout()
plt.show()

mae, rmse, max_error, psnr = calculate_metrics(restored, img)

print("\n--- Denoising Evaluation ---")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"Max Error: {max_error:.4f}")
print(f"PSNR: {psnr:.2f} dB")

noisy_mae, noisy_rmse, noisy_max, noisy_psnr = calculate_metrics(noisy, img)

print("\n--- Before Denoising ---")
print(f"RMSE: {noisy_rmse:.4f}")
print(f"PSNR: {noisy_psnr:.2f} dB")

print("\n--- After Denoising ---")
print(f"RMSE: {rmse:.4f}")
print(f"PSNR: {psnr:.2f} dB")