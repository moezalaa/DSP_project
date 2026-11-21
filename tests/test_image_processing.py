"""Tests for image_signal/processing.py (Student 5: 2D Fourier & image filtering)."""
import numpy as np
import pytest
from PIL import Image
from scipy import signal as sps

from core import filters as core_filters
from image_signal import frequency_filter, image_fft, load_image, spatial_filter

RNG = np.random.default_rng(3)


def make_image(shape=(64, 64)):
    """Smooth background + a bright rectangle + a little texture: enough structure for every filter."""
    r, c = shape
    yy, xx = np.mgrid[:r, :c]
    img = 100 + 50 * np.sin(xx / 9) + 40 * np.cos(yy / 11)
    img[r // 4: r // 2, c // 4: 3 * c // 4] += 60
    img += RNG.normal(0, 2, shape)
    return np.clip(img, 0, 255)


IMG = make_image()
KERNELS = {
    "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], float),
    "sobel_x": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float),
    "sobel_y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], float),
    "prewitt_x": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], float),
    "prewitt_y": np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], float),
    "laplacian": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float),
}


def scipy_conv(img, k):
    return sps.convolve2d(img, k, mode="same", boundary="symm")


def inner(a, m):
    return a[m:-m, m:-m]


# ============================================================ load_image
@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "P", "1"])
def test_load_converts_any_mode_to_2d_grayscale(tmp_path, mode):
    base = Image.fromarray(IMG.astype(np.uint8)).convert("RGB").resize((40, 30))
    im = base.convert(mode)
    p = tmp_path / f"{mode}.png"
    im.save(p)
    x = load_image(p)
    assert x.ndim == 2 and x.shape == (30, 40) and x.dtype == np.float64
    assert x.min() >= 0 and x.max() <= 255
    assert np.array_equal(x, np.asarray(Image.open(p).convert("L"), dtype=float))


def test_load_jpeg_and_string_path(tmp_path):
    p = tmp_path / "a.jpg"
    Image.fromarray(IMG.astype(np.uint8)).save(p)
    assert load_image(str(p)).shape == IMG.shape


def test_load_returns_an_independent_copy(tmp_path):
    p = tmp_path / "a.png"
    Image.fromarray(IMG.astype(np.uint8)).save(p)
    x = load_image(p)
    x[0, 0] = -999                                  # must be writable and not tied to the file
    assert load_image(p)[0, 0] != -999


@pytest.mark.parametrize("bad", ["", None, 123, 4.5])
def test_load_rejects_bad_paths(bad):
    with pytest.raises(ValueError):
        load_image(bad)


def test_load_missing_and_corrupt_files(tmp_path):
    with pytest.raises(ValueError, match="Could not load"):
        load_image(tmp_path / "nope.png")
    junk = tmp_path / "junk.png"
    junk.write_bytes(b"not an image at all")
    with pytest.raises(ValueError, match="Could not load"):
        load_image(junk)


def test_load_16bit_grayscale_keeps_its_contrast(tmp_path):
    ramp = np.linspace(0, 65535, 64 * 64).reshape(64, 64).astype(np.uint16)
    Image.fromarray(ramp).save(tmp_path / "g16.png")
    x = load_image(tmp_path / "g16.png")
    assert x.min() < 10 and x.max() > 245 and 100 < x.mean() < 156     # a spread of grey levels, not all 255


# ============================================================ spatial filters
@pytest.mark.parametrize("name", ["box", "gaussian"])
@pytest.mark.parametrize("size", [3, 5, 7, 9])
def test_blur_matches_scipy_in_the_interior(name, size):
    out = spatial_filter(IMG, name, size=size, sigma=1.4)
    if name == "box":
        k = np.ones((size, size)) / size ** 2
    else:
        ax = np.arange(size) - size // 2
        xx, yy = np.meshgrid(ax, ax)
        k = np.exp(-(xx ** 2 + yy ** 2) / (2 * 1.4 ** 2))
        k /= k.sum()
    m = size // 2
    assert out.shape == IMG.shape and out.dtype == np.float64
    assert np.allclose(inner(out, m), inner(np.clip(scipy_conv(IMG, k), 0, 255), m), atol=1e-9)


@pytest.mark.parametrize("name", ["sharpen", "laplacian"])
def test_clipped_kernels_match_scipy(name):
    out = spatial_filter(IMG, name)
    ref = scipy_conv(IMG, KERNELS[name])
    ref = np.abs(ref) if name == "laplacian" else ref
    assert np.allclose(inner(out, 1), inner(np.clip(ref, 0, 255), 1), atol=1e-9)


@pytest.mark.parametrize("name,kx,ky", [("sobel", "sobel_x", "sobel_y"), ("prewitt", "prewitt_x", "prewitt_y")])
def test_gradient_magnitude_filters_match_scipy(name, kx, ky):
    out = spatial_filter(IMG, name)
    ref = np.clip(np.hypot(scipy_conv(IMG, KERNELS[kx]), scipy_conv(IMG, KERNELS[ky])), 0, 255)
    assert np.allclose(inner(out, 1), inner(ref, 1), atol=1e-9)


@pytest.mark.parametrize("name", ["sobel_x", "sobel_y"])
def test_directional_sobel_keeps_its_sign(name):
    out = spatial_filter(IMG, name)
    ref = scipy_conv(IMG, KERNELS[name])                      # signed, NOT clipped
    assert np.allclose(inner(out, 1), inner(ref, 1), atol=1e-9)
    assert out.min() < 0 < out.max()


def test_blurs_keep_a_constant_image_unchanged_and_edges_vanish_on_it():
    flat = np.full((32, 32), 77.0)
    for name in ("box", "gaussian", "sharpen"):
        assert np.allclose(spatial_filter(flat, name, size=5 if name != "sharpen" else 3), 77.0)
    for name in ("sobel", "sobel_x", "sobel_y", "prewitt", "laplacian"):
        assert np.allclose(spatial_filter(flat, name), 0.0)


def test_outputs_stay_in_the_valid_pixel_range():
    for name in ("box", "gaussian", "sharpen", "sobel", "prewitt", "laplacian"):
        out = spatial_filter(IMG * 1.0, name)
        assert out.min() >= 0 and out.max() <= 255


def test_sobel_x_detects_a_vertical_edge_and_ignores_a_horizontal_one():
    step = np.zeros((20, 20))
    step[:, 10:] = 100
    assert np.abs(spatial_filter(step, "sobel_x")[10, 9:11]).min() > 100
    assert np.allclose(spatial_filter(step, "sobel_y")[5:15, 5:15], 0)


def test_filter_name_is_case_insensitive_and_odd_shapes_work():
    odd = make_image((37, 53))
    assert spatial_filter(odd, "BOX").shape == (37, 53)
    assert np.array_equal(spatial_filter(odd, "Gaussian"), spatial_filter(odd, "gaussian"))


@pytest.mark.parametrize("kw", [dict(size=4), dict(size=0), dict(size=-3), dict(size=3.0), dict(size=True),
                                dict(size="3"), dict(name="median"), dict(name="")])
def test_spatial_filter_validation(kw):
    args = dict(name="box", size=3)
    args.update(kw)
    with pytest.raises(ValueError):
        spatial_filter(IMG, **args)


@pytest.mark.parametrize("sigma", [0, -1, np.nan, np.inf])
def test_gaussian_sigma_validation(sigma):
    with pytest.raises(ValueError, match="sigma"):
        spatial_filter(IMG, "gaussian", size=5, sigma=sigma)


@pytest.mark.parametrize("bad", [np.zeros((4, 4, 3)), np.zeros(5), np.zeros((0, 4)),
                                 np.full((4, 4), np.nan), np.full((4, 4), np.inf)])
def test_images_that_are_not_valid_are_rejected(bad):
    for fn in (lambda: spatial_filter(bad, "box"), lambda: image_fft(bad),
               lambda: frequency_filter(bad, "lowpass", 5)):
        with pytest.raises(ValueError):
            fn()


# ============================================================ 2D Fourier
@pytest.mark.parametrize("shape", [(64, 64), (37, 53), (30, 40), (65, 48)])
def test_image_fft_matches_numpy(shape):
    img = make_image(shape)
    F, logmag, ph = image_fft(img)
    ref = np.fft.fftshift(np.fft.fft2(img))
    assert F.shape == shape and np.allclose(F, ref, atol=1e-6)
    assert np.allclose(logmag, np.log1p(np.abs(ref)), atol=1e-9)
    diff = np.angle(np.exp(1j * (ph - np.angle(ref))))
    assert np.allclose(diff[np.abs(ref) > 1e-6], 0, atol=1e-8)


def test_dc_component_is_at_the_centre_and_equals_the_pixel_sum():
    for shape in [(64, 64), (37, 53)]:
        img = make_image(shape)
        F, logmag, _ = image_fft(img)
        centre = (shape[0] // 2, shape[1] // 2)
        assert np.unravel_index(np.argmax(np.abs(F)), shape) == centre
        assert F[centre].real == pytest.approx(img.sum())


def test_spectrum_of_a_real_image_is_conjugate_symmetric():
    F, _, _ = image_fft(make_image((64, 64)))
    cy, cx = 32, 32
    for i, j in [(1, 0), (0, 5), (3, 7), (10, 10), (20, 31)]:
        assert F[cy + i, cx + j] == pytest.approx(np.conj(F[cy - i, cx - j]), abs=1e-6)


def test_log_magnitude_and_phase_ranges():
    _, logmag, ph = image_fft(IMG)
    assert logmag.min() >= 0 and np.all(np.isfinite(logmag))
    assert ph.min() >= -np.pi and ph.max() <= np.pi


def test_a_pure_sinusoid_gives_two_symmetric_peaks():
    yy, xx = np.mgrid[:64, :64]
    img = 128 + 50 * np.sin(2 * np.pi * (5 * xx / 64 + 3 * yy / 64))
    F, _, _ = image_fft(img)
    mag = np.abs(F)
    mag[32, 32] = 0                                           # drop DC
    top = sorted(zip(*np.unravel_index(np.argsort(mag, axis=None)[-2:], mag.shape)))
    assert top == [(32 - 3, 32 - 5), (32 + 3, 32 + 5)]


# ============================================================ frequency filters
def reference(img, ftype, cutoff, kind, order=2):
    r, c = img.shape
    u, v = np.arange(r) - r // 2, np.arange(c) - c // 2
    d = np.sqrt(u[:, None] ** 2 + v[None, :] ** 2)
    low = {"ideal": (d <= cutoff).astype(float), "gaussian": np.exp(-d ** 2 / (2 * cutoff ** 2)),
           "butterworth": 1 / (1 + (d / cutoff) ** (2 * order))}[kind]
    mask = low if ftype == "lowpass" else 1 - low
    G = np.fft.fftshift(np.fft.fft2(img)) * mask
    return np.clip(np.fft.ifft2(np.fft.ifftshift(G)).real, 0, 255), G


@pytest.mark.parametrize("ftype", ["lowpass", "highpass"])
@pytest.mark.parametrize("kind", ["ideal", "gaussian", "butterworth"])
@pytest.mark.parametrize("shape", [(64, 64), (37, 53)])
def test_frequency_filters_match_a_numpy_reference(ftype, kind, shape):
    img = make_image(shape)
    out, logmag, ph = frequency_filter(img, ftype, 12, kind=kind, order=3)
    ref, G = reference(img, ftype, 12, kind, 3)
    assert out.shape == shape and np.allclose(out, ref, atol=1e-6)
    assert np.allclose(logmag, np.log1p(np.abs(G)), atol=1e-6)
    big = np.abs(G) > 1e-6
    assert np.allclose(np.angle(np.exp(1j * (ph - np.angle(G))))[big], 0, atol=1e-6)


def test_ideal_mask_is_a_circle_not_a_square():
    m = core_filters.freq_mask_2d((65, 65), "lowpass", 10, "ideal")
    assert m[32, 32 + 10] == 1 and m[32 + 10, 32] == 1 and m[32, 32 + 11] == 0
    assert m[32 + 7, 32 + 7] == 1            # distance 9.9 <= 10 (inside)
    assert m[32 + 8, 32 + 8] == 0            # distance 11.3 > 10: a square mask would keep it
    assert m.sum() == pytest.approx(np.pi * 10 ** 2, rel=0.05)


def test_lowpass_blurs_highpass_keeps_edges_and_together_they_rebuild_the_image():
    img = make_image((64, 64))
    low, _, _ = frequency_filter(img, "lowpass", 8)
    high, _, _ = frequency_filter(img, "highpass", 8)
    assert low.std() < img.std()
    assert abs(low.mean() - img.mean()) < 1e-6                  # DC kept by the low-pass
    assert high.min() >= 0                                       # negative values are clipped to 0
    raw_low = np.fft.ifft2(np.fft.ifftshift(reference(img, "lowpass", 8, "ideal")[1])).real
    raw_high = np.fft.ifft2(np.fft.ifftshift(reference(img, "highpass", 8, "ideal")[1])).real
    assert np.allclose(raw_low + raw_high, img, atol=1e-8)


def test_larger_cutoff_keeps_more_detail():
    img = make_image((64, 64))
    errs = [np.mean((frequency_filter(img, "lowpass", c)[0] - img) ** 2) for c in (4, 8, 16, 32)]
    assert errs == sorted(errs, reverse=True)


def test_huge_cutoff_passes_everything_and_tiny_cutoff_leaves_only_the_mean():
    img = make_image((64, 64))
    assert np.allclose(frequency_filter(img, "lowpass", 10_000)[0], img, atol=1e-8)
    flat = frequency_filter(img, "lowpass", 0.5)[0]
    assert np.allclose(flat, img.mean(), atol=1e-8)


def test_order_is_ignored_for_ideal_and_gaussian_but_used_by_butterworth():
    img = make_image((64, 64))
    for kind in ("ideal", "gaussian"):
        assert np.array_equal(frequency_filter(img, "lowpass", 10, kind=kind, order=1)[0],
                              frequency_filter(img, "lowpass", 10, kind=kind, order=9)[0])
    assert not np.allclose(frequency_filter(img, "lowpass", 10, kind="butterworth", order=1)[0],
                           frequency_filter(img, "lowpass", 10, kind="butterworth", order=9)[0])


def test_output_is_clipped_to_pixel_range():
    out, _, _ = frequency_filter(np.where(np.indices((64, 64)).sum(0) % 2, 255.0, 0.0), "highpass", 5)
    assert out.min() >= 0 and out.max() <= 255


@pytest.mark.parametrize("args", [dict(filter_type="bandpass", cutoff=5), dict(filter_type="lowpass", cutoff=0),
                                  dict(filter_type="lowpass", cutoff=-3),
                                  dict(filter_type="lowpass", cutoff=5, kind="notch")])
def test_frequency_filter_validation(args):
    with pytest.raises(ValueError):
        frequency_filter(IMG, **args)


def test_other_deep_formats_keep_their_contrast(tmp_path):
    ints = (np.linspace(0, 70000, 40 * 40).reshape(40, 40)).astype(np.int32)
    Image.fromarray(ints, mode="I").save(tmp_path / "i32.tiff")
    x = load_image(tmp_path / "i32.tiff")
    assert x.min() == 0 and x.max() == 255
    floats = np.linspace(0, 1, 40 * 40).reshape(40, 40).astype(np.float32)
    Image.fromarray(floats, mode="F").save(tmp_path / "f32.tiff")
    y = load_image(tmp_path / "f32.tiff")
    assert y.min() == pytest.approx(0) and y.max() == pytest.approx(255)


def test_normal_8bit_files_are_not_rescaled(tmp_path):
    small = np.array([[0, 10], [20, 30]], dtype=np.uint8)
    Image.fromarray(small).save(tmp_path / "d.png")
    assert np.array_equal(load_image(tmp_path / "d.png"), small)               # a dark 8-bit image stays dark
    dark16 = np.array([[0, 10], [20, 200]], dtype=np.uint16)
    Image.fromarray(dark16).save(tmp_path / "d16.png")
    assert load_image(tmp_path / "d16.png").max() == 200                       # values <= 255 are kept as they are
