"""Tests for image_signal/denoising.py + the metrics (Student 6: periodic noise, notch mask, MSE and PSNR)."""
import numpy as np
import pytest

from image_signal.denoising import (add_periodic_noise, analyze_noisy_image, create_notch_mask,
                                    remove_periodic_noise)
from image_signal.metrics import calculate_metrics, image_metrics, mse, psnr
from scripts.compare_image import calculate_metrics as script_metrics

RNG = np.random.default_rng(4)


def make_image(shape=(128, 128)):
    r, c = shape
    yy, xx = np.mgrid[:r, :c]
    img = 100 + 50 * np.sin(xx / 17) + 40 * np.cos(yy / 23)
    img[r // 4: r // 2, c // 4: 3 * c // 4] += 60
    img += RNG.normal(0, 3, shape)
    return np.clip(img, 0, 255)


IMG = make_image()


def np_spectrum(x):
    return np.fft.fftshift(np.fft.fft2(x))


def top_peaks(img, n=2):
    """Coordinates (row, col) of the n strongest non-DC bins of the centred spectrum."""
    mag = np.abs(np_spectrum(img))
    cy, cx = img.shape[0] // 2, img.shape[1] // 2
    mag[cy, cx] = 0
    idx = np.argsort(mag, axis=None)[-n:]
    return sorted(tuple(int(v) for v in np.unravel_index(i, mag.shape)) for i in idx)


# ============================================================ add_periodic_noise
def test_the_added_noise_is_exactly_a_2d_sinusoid():
    noisy = add_periodic_noise(IMG, amplitude=30, fx=8, fy=6)
    rows, cols = IMG.shape
    yy, xx = np.mgrid[:rows, :cols]
    assert np.allclose(noisy - IMG, 30 * np.sin(2 * np.pi * (8 * xx / cols + 6 * yy / rows)))


def test_noise_is_not_clipped_and_input_is_not_modified():
    before = IMG.copy()
    noisy = add_periodic_noise(IMG, 100, 5, 5)
    assert noisy.max() > 255                                  # NOT clipped: values may leave 0..255
    assert np.array_equal(IMG, before) and noisy.dtype == np.float64 and noisy.shape == IMG.shape


def test_zero_amplitude_is_the_identity():
    assert np.array_equal(add_periodic_noise(IMG, 0, 8, 6), IMG)


@pytest.mark.parametrize("fx,fy", [(8, 6), (1, 1), (20, 3), (5, 40)])
def test_noise_creates_two_spectral_peaks_exactly_at_plus_minus_fx_fy(fx, fy):
    """The brief: the noise frequency is known, so the peaks sit at (centre +- fy, centre +- fx)."""
    flat = np.full(IMG.shape, 128.0)
    noisy = add_periodic_noise(flat, 40, fx, fy)
    cy, cx = IMG.shape[0] // 2, IMG.shape[1] // 2
    assert top_peaks(noisy) == sorted([(cy - fy, cx - fx), (cy + fy, cx + fx)])
    peak = np.abs(np_spectrum(noisy))[cy + fy, cx + fx]
    assert peak == pytest.approx(40 * IMG.size / 2, rel=1e-9)            # A*N/2 for a sine


@pytest.mark.parametrize("shape", [(128, 128), (96, 160), (101, 77)])
def test_peaks_follow_the_image_shape_even_odd_and_rectangular(shape):
    flat = np.full(shape, 128.0)
    noisy = add_periodic_noise(flat, 40, 8, 6)
    cy, cx = shape[0] // 2, shape[1] // 2
    assert top_peaks(noisy) == sorted([(cy - 6, cx - 8), (cy + 6, cx + 8)])


@pytest.mark.parametrize("kw", [dict(amplitude=-1), dict(img=np.zeros((4, 4, 3))), dict(img=np.zeros(5)),
                                dict(img=np.zeros((0, 4)))])
def test_add_periodic_noise_validation(kw):
    args = dict(img=IMG, amplitude=10, fx=8, fy=6)
    args.update(kw)
    with pytest.raises(ValueError):
        add_periodic_noise(**args)


def test_noise_frequency_above_half_the_image_size_is_rejected():
    with pytest.raises(ValueError):
        add_periodic_noise(IMG[:64, :64], 30, fx=40, fy=6)





# ============================================================ analyze_noisy_image
def test_analyze_returns_the_centred_spectrum_and_its_log_magnitude():
    noisy = add_periodic_noise(IMG, 30, 8, 6)
    F, logmag = analyze_noisy_image(noisy)
    assert np.allclose(F, np_spectrum(noisy), atol=1e-5)
    assert np.allclose(logmag, np.log1p(np.abs(F)))
    cy, cx = IMG.shape[0] // 2, IMG.shape[1] // 2
    assert logmag[cy + 6, cx + 8] > np.median(logmag) + 5                # the noise peaks stand out


# ============================================================ notch mask
def test_mask_has_two_symmetric_holes_at_the_noise_frequency():
    mask = create_notch_mask((128, 128), fx=8, fy=6, radius=3)
    cy = cx = 64
    assert mask.shape == (128, 128) and set(np.unique(mask)) == {0.0, 1.0}
    assert mask[cy + 6, cx + 8] == 0 and mask[cy - 6, cx - 8] == 0
    assert mask[cy, cx] == 1                                              # DC (average brightness) is kept
    flipped = mask[::-1, ::-1]                                            # point symmetry about the centre
    assert np.array_equal(mask[1:, 1:], flipped[:-1, :-1])


@pytest.mark.parametrize("radius", [1, 3, 8, 20])
def test_hole_size_follows_the_radius(radius):
    mask = create_notch_mask((128, 128), 30, 25, radius)
    yy, xx = np.ogrid[:128, :128]
    expected = ((yy - (64 + 25)) ** 2 + (xx - (64 + 30)) ** 2 <= radius ** 2)
    assert int((mask == 0).sum()) == 2 * int(expected.sum())
    assert mask[64 + 25, 64 + 30] == 0


def test_mask_leaves_everything_else_untouched():
    mask = create_notch_mask((128, 128), 8, 6, 3)
    assert (mask == 1).sum() == 128 * 128 - (mask == 0).sum() and (mask == 0).sum() == 58


# ============================================================ remove_periodic_noise
def test_restoration_removes_the_noise_and_matches_the_masked_spectrum_exactly():
    noisy = add_periodic_noise(IMG, 30, 8, 6)
    restored, mask = remove_periodic_noise(noisy, 8, 6, 3)
    ref = np.fft.ifft2(np.fft.ifftshift(np_spectrum(noisy) * mask)).real
    assert restored.shape == IMG.shape and np.isrealobj(restored) and np.allclose(restored, ref, atol=1e-6)
    assert mask.shape == IMG.shape


def test_restored_image_is_much_closer_to_the_original_than_the_noisy_one():
    noisy = add_periodic_noise(IMG, 30, 8, 6)
    restored, _ = remove_periodic_noise(noisy, 8, 6, 3)
    _, rmse_n, _, psnr_n = calculate_metrics(noisy, IMG)
    _, rmse_r, _, psnr_r = calculate_metrics(restored, IMG)
    assert rmse_n == pytest.approx(30 / np.sqrt(2), rel=0.01)             # RMS of a sine of amplitude 30
    assert psnr_r - psnr_n > 20                                           # more than 20 dB better
    assert rmse_r ** 2 < 5                                                # MSE of a few grey levels at most
    assert top_peaks(restored)[0] != (64 - 6, 64 - 8)                     # the noise peak is gone


def test_a_wrong_noise_frequency_does_not_restore_anything_useful():
    noisy = add_periodic_noise(IMG, 30, 8, 6)
    good, _ = remove_periodic_noise(noisy, 8, 6, 3)
    wrong, _ = remove_periodic_noise(noisy, 15, 9, 3)
    assert calculate_metrics(wrong, IMG)[1] > 10 * calculate_metrics(good, IMG)[1]


def test_removing_noise_from_a_clean_image_removes_only_two_bins_of_content():
    clean_back, _ = remove_periodic_noise(IMG, 8, 6, 3)
    assert calculate_metrics(clean_back, IMG)[3] > 35                      # still very close: little energy lost


@pytest.mark.parametrize("amp", [10, 30, 60])
def test_restoration_quality_does_not_depend_on_the_noise_strength(amp):
    noisy = add_periodic_noise(IMG, amp, 8, 6)
    restored, _ = remove_periodic_noise(noisy, 8, 6, 3)
    assert calculate_metrics(restored, IMG)[3] > 40


def test_a_too_big_radius_removes_real_image_content():
    noisy = add_periodic_noise(IMG, 30, 8, 6)
    small, _ = remove_periodic_noise(noisy, 8, 6, 1)
    huge, _ = remove_periodic_noise(noisy, 8, 6, 20)
    assert calculate_metrics(huge, IMG)[1] > calculate_metrics(small, IMG)[1]


# ============================================================ metrics (MSE, PSNR)
def test_metrics_values_and_the_mse_relation():
    a = RNG.uniform(0, 255, (40, 40))
    b = np.clip(a + RNG.normal(0, 5, a.shape), 0, 255)
    mae, rmse, mx, psnr = calculate_metrics(b, a)
    mse = np.mean((a - b) ** 2)
    assert rmse ** 2 == pytest.approx(mse)
    assert mae == pytest.approx(np.mean(np.abs(a - b))) and mx == pytest.approx(np.max(np.abs(a - b)))
    assert psnr == pytest.approx(10 * np.log10(255.0 ** 2 / mse))          # the textbook PSNR formula


def test_identical_images_give_infinite_psnr():
    assert calculate_metrics(IMG, IMG.copy())[3] == float("inf")


def test_a_known_psnr_value():
    a = np.zeros((10, 10))
    assert calculate_metrics(a + 1, a)[3] == pytest.approx(20 * np.log10(255), rel=1e-12)   # MSE = 1 -> 48.13 dB


def test_image_metrics_gives_mse_directly():
    a = RNG.uniform(0, 255, (30, 30))
    b = np.clip(a + RNG.normal(0, 4, a.shape), 0, 255)
    r = image_metrics(b, a)
    assert set(r) == {"mae", "mse", "rmse", "max_error", "psnr"}
    assert r["mse"] == pytest.approx(np.mean((a - b) ** 2)) and r["rmse"] == pytest.approx(np.sqrt(r["mse"]))
    assert mse(b, a) == r["mse"] and psnr(b, a) == r["psnr"]
    assert calculate_metrics(b, a) == (r["mae"], r["rmse"], r["max_error"], r["psnr"])
    assert script_metrics is calculate_metrics                    # the script re-exports the package function


def test_metrics_reject_images_of_different_shapes_with_a_clear_message():
    with pytest.raises(ValueError, match="same shape"):
        calculate_metrics(np.zeros((4, 4)), np.zeros((5, 5)))
    with pytest.raises(ValueError, match="same shape"):
        mse(np.zeros((4, 4)), np.zeros((4, 5)))
    with pytest.raises(ValueError):
        image_metrics(np.zeros((0, 0)), np.zeros((0, 0)))


# ============================================================ frequency must be below Nyquist
@pytest.mark.parametrize("fx,fy", [(64, 6), (8, 64), (70, 6), (0, 6), (8, 0), (-3, 6), (8, 200)])
def test_noise_frequencies_outside_the_image_are_rejected(fx, fy):
    with pytest.raises(ValueError, match="alias"):
        add_periodic_noise(IMG, 30, fx, fy)


def test_the_limit_is_strictly_below_half_the_size():
    add_periodic_noise(IMG, 30, 63, 63)                                  # 128 px: 63 < 64 is fine
    with pytest.raises(ValueError):
        add_periodic_noise(IMG, 30, 64, 6)                              # exactly half is not
    small = IMG[:64, :48]
    add_periodic_noise(small, 30, 23, 31)
    with pytest.raises(ValueError):
        add_periodic_noise(small, 30, 24, 6)                            # width 48 -> fx < 24


def test_mask_and_removal_reject_impossible_input():
    with pytest.raises(ValueError):
        create_notch_mask((128, 128), 70, 6, 3)
    with pytest.raises(ValueError, match="radius"):
        create_notch_mask((128, 128), 8, 6, 0)
    with pytest.raises(ValueError):
        create_notch_mask((128,), 8, 6, 3)
    with pytest.raises(ValueError):
        remove_periodic_noise(np.zeros((4, 4, 3)), 1, 1, 1)
    with pytest.raises(ValueError):
        remove_periodic_noise(IMG, 100, 6, 3)
