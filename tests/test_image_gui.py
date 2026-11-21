"""GUI tests for image_signal/gui.py (Hagar's tab). Offscreen: no screen and no file dialogs."""
import re

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("PyQt5")
from PyQt5.QtWidgets import QApplication  # noqa: E402

from image_signal import gui  # noqa: E402
from image_signal.processing import load_image  # noqa: E402
from main import MainWindow  # noqa: E402

app = QApplication.instance() or QApplication([])
RNG = np.random.default_rng(6)


def make_image(shape=(64, 64)):
    r, c = shape
    yy, xx = np.mgrid[:r, :c]
    img = 100 + 50 * np.sin(xx / 9) + 40 * np.cos(yy / 11)
    img[r // 4: r // 2, c // 4: 3 * c // 4] += 60
    return np.clip(img + RNG.normal(0, 2, shape), 0, 255)


@pytest.fixture
def errors(monkeypatch):
    msgs = []
    monkeypatch.setattr(gui, "show_error", lambda parent, message, title="Error": msgs.append((str(title), str(message))))
    return msgs


def pick(monkeypatch, path):
    monkeypatch.setattr(gui.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(path), "")))


def save_to(monkeypatch, path):
    monkeypatch.setattr(gui.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(path), "")))


@pytest.fixture
def png(tmp_path):
    p = tmp_path / "pic.png"
    Image.fromarray(make_image().astype(np.uint8)).save(p)
    return p


@pytest.fixture
def tab(errors, monkeypatch, png):
    t = gui.ImageSignalTab()
    pick(monkeypatch, png)
    t.load_image_file()
    return t


def shown(canvas):
    return len(canvas.ax.images)


def title(canvas):
    return canvas.ax.get_title()


def select(combo, data):
    combo.setCurrentIndex([combo.itemData(i) if combo.itemData(i) is not None else combo.itemText(i)
                           for i in range(combo.count())].index(data))


# ------------------------------------------------------------------ start-up
def test_main_window_loads_the_real_image_tab():
    w = MainWindow()
    assert isinstance(w.tabs.widget(1), gui.ImageSignalTab) and w.tabs.tabText(1) == "Image"


def test_initial_state(errors):
    t = gui.ImageSignalTab()
    assert t.status_label.text() == "Please load an image."
    assert t.original_image is None and t.processed_image is None and t.noisy_image is None
    assert [shown(c) for c in (t.original_canvas, t.processed_canvas, t.magnitude_canvas, t.phase_canvas)] == [0] * 4


def test_every_action_asks_for_an_image_first(errors):
    t = gui.ImageSignalTab()
    for action in (t.apply_spatial_filter, t.analyze_fft, t.apply_frequency_filter, t.add_noise, t.reset_image):
        action()
    t.remove_noise()
    t.save_processed_image()
    messages = [m for _, m in errors]
    assert messages.count("Please load an image first.") == 5
    assert "Please add periodic noise first." in messages and "No processed image to save." in messages


# ------------------------------------------------------------------ loading
def test_load_shows_the_original_only(tab, png):
    assert np.array_equal(tab.original_image, load_image(png))
    assert shown(tab.original_canvas) == 1 and title(tab.original_canvas) == "Original Image"
    assert [shown(c) for c in (tab.processed_canvas, tab.magnitude_canvas, tab.phase_canvas)] == [0, 0, 0]
    assert tab.status_label.text() == "Image loaded successfully."


def test_load_cancelled_and_load_failure(tab, errors, monkeypatch, tmp_path):
    before = tab.original_image.copy()
    pick(monkeypatch, "")
    tab.load_image_file()
    assert np.array_equal(tab.original_image, before) and not errors
    bad = tmp_path / "junk.png"
    bad.write_bytes(b"junk")
    pick(monkeypatch, bad)
    tab.load_image_file()
    assert errors and errors[-1][0] == "Image Loading Error" and "Could not load" in errors[-1][1]
    assert np.array_equal(tab.original_image, before)


# ------------------------------------------------------------------ spatial filters
@pytest.mark.parametrize("name", ["box", "gaussian", "sharpen", "sobel_x", "sobel_y", "sobel", "prewitt", "laplacian"])
@pytest.mark.parametrize("size", ["3", "7"])
def test_every_spatial_filter_runs_from_the_combo(tab, errors, name, size):
    select(tab.filter_combo, name)
    tab.kernel_size.setCurrentText(size)
    tab.apply_spatial_filter()
    assert not errors
    assert tab.processed_image.shape == tab.original_image.shape
    assert shown(tab.processed_canvas) == 1 and title(tab.processed_canvas) == f"Processed Image - {name}"
    assert [shown(tab.magnitude_canvas), shown(tab.phase_canvas)] == [0, 0]
    assert tab.status_label.text() == f"{name.capitalize()} filter applied successfully."


def test_gaussian_uses_the_sigma_spinbox(tab):
    select(tab.filter_combo, "gaussian")
    tab.kernel_size.setCurrentText("9")
    tab.sigma.setValue(0.5)
    tab.apply_spatial_filter()
    narrow = tab.processed_image.copy()
    tab.sigma.setValue(4.0)
    tab.apply_spatial_filter()
    assert narrow.std() > tab.processed_image.std()                       # wider sigma blurs more


# ------------------------------------------------------------------ FFT + frequency filters (Student 5)
def test_analyze_fft_shows_log_magnitude_and_phase(tab):
    tab.analyze_fft()
    assert title(tab.processed_canvas) == "FFT - Log Magnitude" and title(tab.phase_canvas) == "FFT - Phase"
    assert shown(tab.processed_canvas) == 1 and shown(tab.phase_canvas) == 1 and shown(tab.magnitude_canvas) == 0
    assert tab.status_label.text() == "FFT analysis completed successfully."
    assert tab.phase_canvas.ax.images[0].get_cmap().name == "twilight"


def test_the_order_box_is_enabled_only_for_butterworth(tab):
    for kind, enabled in [("ideal", False), ("gaussian", False), ("butterworth", True)]:
        tab.freq_kind.setCurrentText(kind)
        assert tab.order.isEnabled() is enabled


@pytest.mark.parametrize("ftype", ["lowpass", "highpass"])
@pytest.mark.parametrize("kind", ["ideal", "gaussian", "butterworth"])
def test_every_frequency_filter_fills_the_three_result_canvases(tab, errors, ftype, kind):
    select(tab.freq_type, ftype)
    tab.freq_kind.setCurrentText(kind)
    tab.cutoff.setValue(12)
    tab.apply_frequency_filter()
    assert not errors
    assert title(tab.processed_canvas) == f"Frequency Filter - {ftype}"
    assert title(tab.magnitude_canvas) == "Filtered Log Magnitude" and title(tab.phase_canvas) == "Filtered Phase"
    assert [shown(c) for c in (tab.processed_canvas, tab.magnitude_canvas, tab.phase_canvas)] == [1, 1, 1]
    assert tab.status_label.text() == f"{kind.capitalize()} {ftype} applied successfully."


def test_lowpass_via_the_gui_is_blurrier_than_the_original(tab):
    select(tab.freq_type, "lowpass")
    tab.cutoff.setValue(5)
    tab.apply_frequency_filter()
    assert tab.processed_image.std() < tab.original_image.std()


def test_an_invalid_cutoff_is_reported_not_raised(tab, errors, monkeypatch):
    monkeypatch.setattr(tab.cutoff, "value", lambda: -4.0)
    tab.apply_frequency_filter()
    assert errors and errors[-1][0] == "Frequency Filter Error"


# ------------------------------------------------------------------ periodic noise + restoration (Student 6)
def test_add_noise_shows_the_noisy_image_and_its_spectrum(tab):
    tab.noise_amplitude.setValue(30)
    tab.noise_fx.setValue(8)
    tab.noise_fy.setValue(6)
    tab.add_noise()
    assert np.allclose(tab.noisy_image - tab.original_image,
                       30 * np.sin(2 * np.pi * (8 * np.arange(64)[None, :] / 64 + 6 * np.arange(64)[:, None] / 64)))
    assert tab.processed_image is tab.noisy_image
    assert title(tab.processed_canvas) == "Image with Periodic Noise"
    assert title(tab.magnitude_canvas) == "Noisy Image - FFT Spectrum"
    assert tab.status_label.text() == "Periodic noise added successfully."





def test_the_notch_radius_changes_the_result(tab):
    tab.add_noise()
    tab.notch_radius.setValue(1)
    tab.remove_noise()
    small = tab.processed_image.copy()
    tab.notch_radius.setValue(20)
    tab.remove_noise()
    assert not np.allclose(small, tab.processed_image)


def test_a_wrong_frequency_in_the_removal_step_gives_a_worse_result(tab):
    tab.add_noise()
    tab.remove_noise()
    good = float(re.search(r"PSNR: ([\d.]+)", tab.status_label.text()).group(1))
    tab.noise_fx.setValue(15)
    tab.remove_noise()
    bad = float(re.search(r"PSNR: ([\d.]+)", tab.status_label.text()).group(1))
    assert good > bad + 10

















# ------------------------------------------------------------------ reset / save
def test_reset_clears_the_results_and_keeps_the_original(tab):
    tab.analyze_fft()
    tab.reset_image()
    assert tab.processed_image is None and shown(tab.original_canvas) == 1
    assert [shown(c) for c in (tab.processed_canvas, tab.magnitude_canvas, tab.phase_canvas)] == [0, 0, 0]
    assert tab.status_label.text() == "Image reset successfully."





def test_save_jpg_cancel_and_failure(tab, errors, monkeypatch, tmp_path):
    tab.analyze_fft()
    tab.processed_image = tab.original_image
    save_to(monkeypatch, tmp_path / "a.jpg")
    tab.save_processed_image()
    assert (tmp_path / "a.jpg").exists() and not errors
    save_to(monkeypatch, "")
    tab.save_processed_image()
    assert not errors
    save_to(monkeypatch, tmp_path / "no_such_dir" / "x.png")
    tab.save_processed_image()
    assert errors and errors[-1][0] == "Save Error"





# ------------------------------------------------------------------ whole brief workflow through the GUI


# ------------------------------------------------------------------ Student 6: restoration, MSE, full comparison
STATUS = re.compile(r"Restored \| MSE: ([\d.]+) \| RMSE: ([\d.]+) \| PSNR: ([\d.]+) dB  "
                    r"\(noisy: MSE ([\d.]+), PSNR ([\d.]+) dB\)")


def test_remove_noise_reports_mse_rmse_psnr_before_and_after(tab, errors):
    tab.add_noise()
    tab.remove_noise()
    assert not errors
    m = STATUS.fullmatch(tab.status_label.text())
    assert m, tab.status_label.text()
    mse_after, rmse_after, psnr_after, mse_before, psnr_before = map(float, m.groups())
    assert rmse_after ** 2 == pytest.approx(mse_after, rel=0.05, abs=0.02)
    assert psnr_after > psnr_before + 15 and mse_after < mse_before / 30
    assert mse_before == pytest.approx(450, rel=0.02)                       # amplitude 30 -> MSE = 30^2 / 2


def test_original_noisy_and_restored_are_all_visible_together(tab):
    tab.add_noise()
    assert not tab.noisy_canvas.isVisibleTo(tab) and not tab.mask_canvas.isVisibleTo(tab)
    tab.remove_noise()
    canvases = [tab.original_canvas, tab.processed_canvas, tab.noisy_canvas,
                tab.magnitude_canvas, tab.phase_canvas, tab.mask_canvas]
    assert all(c.isVisibleTo(tab) and shown(c) == 1 for c in canvases)
    assert [title(c) for c in canvases] == ["Original Image", "Restored Image", "Noisy Image",
                                            "Restored Image - FFT Spectrum", "Noisy Image - FFT Spectrum", "Notch Mask"]
    assert np.array_equal(np.asarray(tab.noisy_canvas.ax.images[0].get_array()), tab.noisy_image)
    mask = np.asarray(tab.mask_canvas.ax.images[0].get_array())
    assert set(np.unique(mask)) == {0.0, 1.0} and mask.shape == tab.original_image.shape


def test_the_third_column_disappears_when_another_operation_runs(tab, monkeypatch, png):
    for action in (tab.analyze_fft, tab.apply_spatial_filter, tab.apply_frequency_filter,
                   tab.add_noise, tab.reset_image):
        tab.add_noise()
        tab.remove_noise()
        assert tab.noisy_canvas.isVisibleTo(tab)
        action()
        assert not tab.noisy_canvas.isVisibleTo(tab) and not tab.mask_canvas.isVisibleTo(tab)
        assert shown(tab.noisy_canvas) == 0 and shown(tab.mask_canvas) == 0
    tab.add_noise()
    tab.remove_noise()
    pick(monkeypatch, png)
    tab.load_image_file()
    assert not tab.noisy_canvas.isVisibleTo(tab)


def test_the_restored_spectrum_has_the_two_noise_peaks_removed(tab):
    tab.add_noise()
    tab.remove_noise()
    noisy_spec = np.asarray(tab.phase_canvas.ax.images[0].get_array())
    restored_spec = np.asarray(tab.magnitude_canvas.ax.images[0].get_array())
    cy = cx = 32
    for sign in (1, -1):
        y, x = cy + sign * 6, cx + sign * 8
        assert noisy_spec[y, x] > restored_spec[y, x] + 3                    # peak clearly lower after the notch


def test_loading_a_new_image_forgets_the_previous_noisy_one(tab, errors, monkeypatch, tmp_path):
    tab.add_noise()
    other = tmp_path / "other.png"
    Image.fromarray(make_image((48, 80)).astype(np.uint8)).save(other)
    pick(monkeypatch, other)
    tab.load_image_file()
    assert tab.noisy_image is None
    tab.remove_noise()
    assert errors and errors[-1][1] == "Please add periodic noise first."


def test_noise_frequency_limits_follow_the_image_size(tab, errors, monkeypatch, tmp_path):
    assert (tab.noise_fx.maximum(), tab.noise_fy.maximum()) == (31, 31)      # 64 x 64 image
    tab.noise_fx.setValue(100)                                              # typed value is clamped
    assert tab.noise_fx.value() == 31
    tab.add_noise()
    assert not errors
    rect = tmp_path / "rect.png"
    Image.fromarray(make_image((48, 80)).astype(np.uint8)).save(rect)
    pick(monkeypatch, rect)
    tab.load_image_file()
    assert (tab.noise_fx.maximum(), tab.noise_fy.maximum()) == (39, 23)      # width 80, height 48
    tiny = tmp_path / "tiny.png"
    Image.fromarray(make_image((3, 3)).astype(np.uint8)).save(tiny)
    pick(monkeypatch, tiny)
    tab.load_image_file()
    assert tab.noise_fx.maximum() >= 1


def test_core_errors_about_frequency_are_still_reported_by_the_gui(tab, errors, monkeypatch):
    monkeypatch.setattr(tab.noise_fx, "value", lambda: 60)
    tab.add_noise()
    assert errors and errors[-1][0] == "Noise Error" and "alias" in errors[-1][1]


def test_different_frequencies_for_noise_and_removal(tab):
    tab.noise_fx.setValue(8)
    tab.add_noise()
    tab.remove_noise()
    good = float(STATUS.fullmatch(tab.status_label.text()).group(3))
    tab.noise_fx.setValue(15)
    tab.remove_noise()
    bad = float(STATUS.fullmatch(tab.status_label.text()).group(3))
    assert good > bad + 10


# ------------------------------------------------------------------ saving
def test_save_writes_a_readable_png_equal_to_the_result(tab, errors, monkeypatch, tmp_path):
    select(tab.filter_combo, "box")
    tab.apply_spatial_filter()
    out = tmp_path / "result.png"
    save_to(monkeypatch, out)
    tab.save_processed_image()
    assert not errors and tab.status_label.text() == "Image saved successfully."
    saved = np.asarray(Image.open(out))
    assert saved.shape == tab.processed_image.shape
    assert np.max(np.abs(saved - np.clip(tab.processed_image, 0, 255))) <= 0.5 + 1e-9      # rounded, not truncated


def test_saving_a_signed_sobel_rescales_instead_of_losing_the_negative_half(tab, monkeypatch, tmp_path):
    select(tab.filter_combo, "sobel_x")
    tab.apply_spatial_filter()
    assert tab.processed_image.min() < 0
    save_to(monkeypatch, tmp_path / "s.png")
    tab.save_processed_image()
    saved = np.asarray(Image.open(tmp_path / "s.png"))
    assert saved.min() == 0 and saved.max() == 255 and "rescaled" in tab.status_label.text()
    zero_level = round(-tab.processed_image.min() / (tab.processed_image.max() - tab.processed_image.min()) * 255)
    flat = np.abs(tab.processed_image) < 1e-9
    if flat.any():
        assert abs(int(np.median(saved[flat])) - zero_level) <= 1                      # "no edge" maps to mid grey


# ------------------------------------------------------------------ whole brief workflow through the GUI
def test_student_5_and_6_workflow(tab, errors):
    tab.analyze_fft()                                                    # 2D spectrum (fft2 + fftshift)
    select(tab.freq_type, "lowpass"); tab.freq_kind.setCurrentText("ideal"); tab.cutoff.setValue(10)
    tab.apply_frequency_filter()                                         # ideal circular low-pass
    select(tab.freq_type, "highpass")
    tab.apply_frequency_filter()                                         # ideal circular high-pass
    tab.add_noise()                                                      # periodic noise + its spectrum
    assert title(tab.magnitude_canvas) == "Noisy Image - FFT Spectrum"
    tab.remove_noise()                                                   # notch mask + inverse FFT + MSE/PSNR
    assert not errors and STATUS.fullmatch(tab.status_label.text())

