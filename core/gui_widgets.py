"""Reusable PyQt5 widgets: matplotlib canvases + a labeled slider/spinbox.

Everyone's GUI imports from here, so all tabs look and behave the same.
"""
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QDoubleSpinBox, QHBoxLayout, QLabel, QMessageBox,
                             QSlider, QWidget)


class PlotCanvas(FigureCanvasQTAgg):
    """A matplotlib figure with `rows` x `cols` line plots.

    canvas = PlotCanvas(rows=3)
    canvas.plot(t, x, index=0, title="Signal", xlabel="Time (s)")
    canvas.plot(freqs, mag, index=1, title="Magnitude", stem=True)
    """

    def __init__(self, rows=1, cols=1, parent=None):
        self.figure = Figure(tight_layout=True)
        super().__init__(self.figure)
        self.setParent(parent)
        self.axes = self.figure.subplots(rows, cols, squeeze=False).ravel()

    def plot(self, x, y, index=0, title="", xlabel="", ylabel="", stem=False,
             overlay=None, labels=("", ""), marker=None):
        """Draw y(x) on subplot `index`. `overlay=(x2, y2)` draws a second curve.

        marker: e.g. "o" to draw a dot on every point. Use it when y contains NaN gaps,
        so isolated points (which have no neighbour to connect to) are still visible.
        """
        ax = self.axes[index]
        ax.clear()
        if stem:
            ax.stem(x, y, basefmt=" ")
        else:
            ax.plot(x, y, label=labels[0], marker=marker, markersize=4)
        if overlay is not None:
            ax.plot(overlay[0], overlay[1], label=labels[1], alpha=0.8,
                    marker=marker, markersize=3)
        if any(labels):
            ax.legend(loc="upper right")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        self.draw_idle()

    def clear(self):
        for ax in self.axes:
            ax.clear()
        self.draw_idle()


class ImageCanvas(FigureCanvasQTAgg):
    """A matplotlib figure that shows one image (no axes ticks)."""

    def __init__(self, title="", parent=None):
        self.figure = Figure(tight_layout=True)
        super().__init__(self.figure)
        self.setParent(parent)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title(title)
        self.ax.axis("off")

    def show_image(self, img, title="", cmap="gray"):
        self.ax.clear()
        self.ax.imshow(np.asarray(img), cmap=cmap)
        self.ax.set_title(title)
        self.ax.axis("off")
        self.draw_idle()

    def clear(self):
        self.ax.clear()
        self.ax.axis("off")
        self.draw_idle()


class LabeledSlider(QWidget):
    """label + slider + spinbox kept in sync. Emits valueChanged(float).

    s = LabeledSlider("Frequency (Hz)", 1, 500, 50, step=1, decimals=0)
    s.valueChanged.connect(my_callback);   s.value()
    """
    valueChanged = pyqtSignal(float)
    _STEPS = 1000

    def __init__(self, label, minimum, maximum, value, step=1.0, decimals=2, parent=None):
        super().__init__(parent)
        self._min, self._max = float(minimum), float(maximum)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(label)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, self._STEPS)
        self.spin = QDoubleSpinBox()
        self.spin.setRange(self._min, self._max)
        self.spin.setSingleStep(step)
        self.spin.setDecimals(decimals)
        layout.addWidget(self.label, 2)
        layout.addWidget(self.slider, 3)
        layout.addWidget(self.spin, 1)
        self.setValue(value)
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)

    def value(self) -> float:
        return self.spin.value()

    def setValue(self, v):
        self.spin.setValue(float(v))
        self._sync_slider()

    def _sync_slider(self):
        frac = (self.spin.value() - self._min) / (self._max - self._min)
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(frac * self._STEPS)))
        self.slider.blockSignals(False)

    def _from_slider(self, pos):
        v = self._min + pos / self._STEPS * (self._max - self._min)
        self.spin.blockSignals(True)
        self.spin.setValue(v)
        self.spin.blockSignals(False)
        self.valueChanged.emit(self.spin.value())

    def _from_spin(self, v):
        self._sync_slider()
        self.valueChanged.emit(v)


def show_error(parent, message, title="Error"):
    """Standard error popup: the GUI catches ValueError from core and calls this."""
    QMessageBox.warning(parent, title, str(message))
