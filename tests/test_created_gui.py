import numpy as np
import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtWidgets import QApplication  # noqa: E402

from created_signal.gui import CreatedSignalTab  # noqa: E402
from main import MainWindow  # noqa: E402

app = QApplication.instance() or QApplication([])


@pytest.fixture
def tab():
    t = CreatedSignalTab()
    t.auto.setChecked(False)
    return t


def select_filter(tab, key):
    keys = [tab.filter_type.itemData(i) for i in range(tab.filter_type.count())]
    tab.filter_type.setCurrentIndex(keys.index(key))


def test_main_window_loads_real_tab():
    w = MainWindow()
    assert isinstance(w.tabs.widget(0), CreatedSignalTab)


def test_default_state_plots(tab):
    tab.update_plots()
    assert tab.result is not None and tab.status.text() == "OK"
    assert len(tab.result["t"]) == 1000


def test_lowpass_removes_second_tone(tab):
    select_filter(tab, "lowpass")
    tab.cutoff.setValue(20)
    tab.update_plots()
    r = tab.result
    assert r["mag_in"][40] == pytest.approx(0.5, abs=1e-9)
    assert r["mag_out"][40] < 1e-9 and r["mag_out"][5] == pytest.approx(1.0, abs=1e-9)


def test_filter_controls_visibility(tab):
    select_filter(tab, "bandpass")
    assert tab.cutoff_high.isVisibleTo(tab) and not tab.kernel_row.isVisibleTo(tab)
    select_filter(tab, "custom")
    assert tab.kernel_row.isVisibleTo(tab) and not tab.cutoff.isVisibleTo(tab)
    select_filter(tab, "none")
    assert not tab.size.isVisibleTo(tab)


def test_noise_and_new_noise(tab):
    tab.noise_box.setChecked(True)
    tab.update_plots()
    first = tab.result["input"].copy()
    tab.new_noise.click()
    assert not np.allclose(first, tab.result["input"])
    assert not np.allclose(tab.result["input"], tab.result["clean"])


def test_nyquist_warning_shown(tab):
    tab.fs.setValue(500)
    tab.components[1].freq.setValue(400)
    tab.update_plots()
    assert "alias" in tab.status.text()


def test_all_components_off_shows_error_not_crash(tab):
    for c in tab.components:
        c.setChecked(False)
    tab.update_plots()
    assert "at least one" in tab.status.text()


def test_bad_custom_kernel_shows_error(tab):
    select_filter(tab, "custom")
    tab.kernel_text.setText("1, abc")
    tab.update_plots()
    assert "numbers" in tab.status.text()
    tab.kernel_text.setText("1 2 1")
    tab.update_plots()
    assert tab.status.text() == "OK"


def test_comparison_tab_fills_table_and_matches(tab):
    select_filter(tab, "gaussian")
    tab.right.setCurrentIndex(1)
    tab.update_plots()
    assert tab.table.rowCount() == 9
    assert tab.comparison["all_ok"] and "match" in tab.cmp_summary.text()
    assert tab.table.item(5, 0).text().startswith("Convolution filter: gaussian")
    assert tab.table.item(8, 0).text().startswith("Frequency filter: lowpass")
    assert tab.table.item(0, 6).text() == "✓"


def test_comparison_is_lazy(tab):
    tab.update_plots()
    assert tab.comparison is None                    # signals tab visible: nothing computed
    tab.right.setCurrentIndex(1)
    assert tab.comparison is not None


def test_export_csv(tab, tmp_path):
    tab.update_plots()
    path = tmp_path / "out.csv"
    tab.export_csv(str(path))
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    assert data.shape == (1000, 4)
    assert np.allclose(data[:, 2], tab.result["input"])


def test_export_before_any_result_raises():
    t = CreatedSignalTab.__new__(CreatedSignalTab)
    t.result = None
    with pytest.raises(ValueError):
        CreatedSignalTab.export_csv(t, "x.csv")


def test_reset_restores_defaults(tab):
    tab.fs.setValue(5000)
    select_filter(tab, "notch")
    tab.reset()
    assert tab.fs.value() == 1000 and tab.filter_type.currentData() == "none"


def _phase_points(tab):
    """Number of finite phase points drawn for (input, output)."""
    lines = tab.canvas.axes[2].get_lines()
    return [int(np.sum(np.isfinite(l.get_ydata()))) for l in lines]


@pytest.mark.parametrize("ftype,noise", [("none", False), ("lowpass", False),
                                         ("lowpass", True), ("highpass", False),
                                         ("notch", False), ("difference", True)])
def test_phase_plot_always_has_points(tab, ftype, noise):
    select_filter(tab, ftype)
    tab.noise_box.setChecked(noise)
    tab.update_plots()
    pts = _phase_points(tab)
    assert len(pts) == 2 and min(pts) >= 1
    assert all(l.get_marker() == "o" for l in tab.canvas.axes[2].get_lines())


def test_phase_plot_when_filter_removes_everything(tab):
    select_filter(tab, "highpass")
    tab.cutoff.setValue(5000)                       # output is (almost) all zeros
    tab.update_plots()
    assert min(_phase_points(tab)) >= 1


def test_phase_threshold_slider(tab):
    tab.noise_box.setChecked(True)
    tab.phase_thr.setValue(0)
    tab.update_plots()
    all_bins = _phase_points(tab)[0]
    tab.phase_thr.setValue(50)
    tab.update_plots()
    assert 1 <= _phase_points(tab)[0] < all_bins


def test_difference_equation_filter_and_instability_warning(tab):
    select_filter(tab, "first_order")
    assert tab.de_a.isVisibleTo(tab) and tab.de_b.isVisibleTo(tab) and not tab.cutoff.isVisibleTo(tab)
    tab.update_plots()
    assert tab.status.text() == "OK"
    tab.de_a.setValue(1.1)
    tab.update_plots()
    assert "UNSTABLE" in tab.status.text()


def test_dtft_tab_runs_and_draws(tab):
    tab.update_plots()
    tab.right.setCurrentIndex(2)
    tab.dtft_n.setValue(128)
    tab.dtft_k.setValue(64)
    tab.run_dtft()
    mag_line = tab.dtft_canvas.axes[0].get_lines()[0]
    assert len(mag_line.get_xdata()) == 64
    assert len(tab.dtft_canvas.axes[0].get_lines()) == 2          # DTFT curve + DFT samples
    assert "128 samples" in tab.dtft_info.text()
    assert min(int(np.sum(np.isfinite(l.get_ydata()))) for l in tab.dtft_canvas.axes[1].get_lines()) >= 1


def test_convolution_tab_shows_padding_effect(tab):
    tab.update_plots()
    tab.right.setCurrentIndex(3)
    assert "CIRCULAR" in tab.conv_info.text()
    assert len(tab.conv_canvas.axes[0].get_lines()) == 2


def test_all_extra_tabs_update_when_a_parameter_changes(tab):
    tab.update_plots()
    for idx in (1, 2, 3):
        tab.right.setCurrentIndex(idx)
        tab.components[0].freq.setValue(10 + idx)
        tab.update_plots()                                         # must not raise
    assert tab.right.count() == 4
