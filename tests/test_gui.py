"""Smoke tests: widgets and main window can be created (offscreen, no display needed)."""
import numpy as np
import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtWidgets import QApplication  # noqa: E402

from core.gui_widgets import ImageCanvas, LabeledSlider, PlotCanvas  # noqa: E402
from main import MainWindow  # noqa: E402

app = QApplication.instance() or QApplication([])


def test_plot_canvas():
    c = PlotCanvas(rows=3)
    t = np.linspace(0, 1, 100)
    c.plot(t, np.sin(t), index=0, title="x")
    c.plot(t, np.cos(t), index=1, stem=True)
    c.plot(t, t, index=2, overlay=(t, t ** 2), labels=("a", "b"))
    c.clear()


def test_image_canvas():
    c = ImageCanvas("img")
    c.show_image(np.random.rand(10, 10), "t")
    c.clear()


def test_labeled_slider_sync():
    s = LabeledSlider("f", 0, 100, 25, step=1, decimals=0)
    got = []
    s.valueChanged.connect(got.append)
    s.setValue(50)
    assert s.value() == 50
    s.slider.setValue(1000)
    assert s.value() == 100 and got


def test_main_window_has_three_tabs():
    w = MainWindow()
    assert w.tabs.count() == 3
    assert [w.tabs.tabText(i) for i in range(3)] == ["Created Signal", "Image", "Speech"]
