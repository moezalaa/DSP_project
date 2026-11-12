from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QLineEdit,
    QCheckBox,
    QFileDialog,
    QMessageBox,
    QSplitter,
    QScrollArea,
    QSizePolicy,
)

from core.gui_widgets import PlotCanvas, show_error

@dataclass
class SignalRecord:
    """A generated signal and the parameters needed to interpret it."""

    time: np.ndarray
    samples: np.ndarray
    fs: float
    name: str

class CreatedSignalTab(QWidget):
    """Created Signal tab loaded by ``main.py``."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.signal_records: list[SignalRecord] = []
        self.reference_signal: Optional[np.ndarray] = None
        self.current_signal: Optional[np.ndarray] = None
        self.time: Optional[np.ndarray] = None
        self.fs: Optional[float] = None
        self.current_name = "No signal"

        self._build_ui()
        self._set_signal_controls_enabled(False)
        self._set_status("Generate a signal to get started.")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # Controls are scrollable so the tab remains usable on smaller screens.
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(4, 4, 8, 4)
        controls_layout.setSpacing(10)

        controls_layout.addWidget(self._build_generation_group())
        controls_layout.addWidget(self._build_collection_group())
        controls_layout.addWidget(self._build_noise_group())
        controls_layout.addWidget(self._build_convolution_group())
        controls_layout.addWidget(self._build_frequency_filter_group())
        controls_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls_widget)
        scroll.setMinimumWidth(300)
        scroll.setMaximumWidth(390)
        splitter.addWidget(scroll)

        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        plot_layout.setContentsMargins(4, 4, 4, 4)

        header = QHBoxLayout()
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #666;")
        header.addWidget(self.status_label, 1)

        self.compare_check = QCheckBox("Compare with original")
        self.compare_check.setChecked(True)
        self.compare_check.setEnabled(False)
        self.compare_check.toggled.connect(self._refresh_plots)
        header.addWidget(self.compare_check)

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_signal)
        self.reset_button.setEnabled(False)
        header.addWidget(self.reset_button)

        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_signal)
        self.save_button.setEnabled(False)
        header.addWidget(self.save_button)

        plot_layout.addLayout(header)

        self.canvas = PlotCanvas(rows=3, cols=1)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        plot_layout.addWidget(self.canvas, 1)

        splitter.addWidget(plot_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 850])

    def _build_generation_group(self) -> QGroupBox:
        group = QGroupBox("Generate signal")
        form = QFormLayout(group)

        self.signal_kind = QComboBox()
        self.signal_kind.addItems(
            ["sine", "cosine", "square", "triangle", "sawtooth"]
        )
        form.addRow("Waveform", self.signal_kind)

        self.amplitude = self._double_spin(0.0, 1_000_000.0, 1.0, 3)
        form.addRow("Amplitude", self.amplitude)

        self.frequency = self._double_spin(0.01, 1_000_000.0, 5.0, 2)
        form.addRow("Frequency (Hz)", self.frequency)

        self.phase = self._double_spin(-1000.0, 1000.0, 0.0, 3)
        form.addRow("Phase (rad)", self.phase)

        self.sampling_rate = self._double_spin(1.0, 10_000_000.0, 1000.0, 1)
        form.addRow("Sampling rate (Hz)", self.sampling_rate)

        self.duration = self._double_spin(0.001, 100_000.0, 1.0, 3)
        form.addRow("Duration (s)", self.duration)

        self.generate_button = QPushButton("Generate")
        self.generate_button.clicked.connect(self.generate_signal)
        form.addRow(self.generate_button)

        return group

    def _build_collection_group(self) -> QGroupBox:
        group = QGroupBox("Combine signals")
        layout = QVBoxLayout(group)

        self.collection_label = QLabel("No generated signals yet.")
        self.collection_label.setWordWrap(True)
        layout.addWidget(self.collection_label)

        buttons = QHBoxLayout()
        self.add_signal_button = QPushButton("Add current")
        self.add_signal_button.clicked.connect(self.add_current_to_collection)
        self.add_signal_button.setEnabled(False)
        buttons.addWidget(self.add_signal_button)

        self.combine_button = QPushButton("Combine all")
        self.combine_button.clicked.connect(self.combine_signals)
        self.combine_button.setEnabled(False)
        buttons.addWidget(self.combine_button)
        layout.addLayout(buttons)

        self.clear_collection_button = QPushButton("Clear collection")
        self.clear_collection_button.clicked.connect(self.clear_collection)
        self.clear_collection_button.setEnabled(False)
        layout.addWidget(self.clear_collection_button)

        return group

    def _build_noise_group(self) -> QGroupBox:
        group = QGroupBox("Add Gaussian noise")
        form = QFormLayout(group)

        self.noise_snr = self._double_spin(-20.0, 100.0, 20.0, 1)
        form.addRow("Target SNR (dB)", self.noise_snr)

        self.noise_button = QPushButton("Add noise to current signal")
        self.noise_button.clicked.connect(self.add_noise)
        form.addRow(self.noise_button)

        return group

    def _build_convolution_group(self) -> QGroupBox:
        group = QGroupBox("Convolution filter")
        form = QFormLayout(group)

        self.conv_kind = QComboBox()
        self.conv_kind.addItems(["moving_average", "difference", "custom"])
        self.conv_kind.currentTextChanged.connect(self._update_custom_kernel_state)
        form.addRow("Kernel type", self.conv_kind)

        self.conv_size = QSpinBox()
        self.conv_size.setRange(1, 10001)
        self.conv_size.setSingleStep(2)
        self.conv_size.setValue(5)
        form.addRow("Kernel size", self.conv_size)

        self.custom_kernel = QLineEdit("1, 0, -1")
        self.custom_kernel.setPlaceholderText("e.g. 0.25, 0.5, 0.25")
        form.addRow("Custom kernel", self.custom_kernel)

        self.conv_button = QPushButton("Apply convolution")
        self.conv_button.clicked.connect(self.apply_convolution)
        form.addRow(self.conv_button)

        self._update_custom_kernel_state()
        return group

    def _build_frequency_filter_group(self) -> QGroupBox:
        group = QGroupBox("Frequency-domain filter")
        form = QFormLayout(group)

        self.freq_filter_kind = QComboBox()
        self.freq_filter_kind.addItems(["lowpass", "highpass", "bandpass", "notch"])
        self.freq_filter_kind.currentTextChanged.connect(
            self._update_frequency_filter_state
        )
        form.addRow("Filter", self.freq_filter_kind)

        self.cutoff_low = self._double_spin(0.0, 10_000_000.0, 20.0, 2)
        form.addRow("Cutoff / lower (Hz)", self.cutoff_low)

        self.cutoff_high = self._double_spin(0.0, 10_000_000.0, 100.0, 2)
        form.addRow("Upper cutoff (Hz)", self.cutoff_high)

        self.notch_width = self._double_spin(0.01, 10_000_000.0, 2.0, 2)
        form.addRow("Notch width (Hz)", self.notch_width)

        self.filter_order = QSpinBox()
        self.filter_order.setRange(1, 20)
        self.filter_order.setValue(2)
        form.addRow("Butterworth order", self.filter_order)

        self.ideal_filter = QCheckBox("Use ideal filter")
        self.ideal_filter.setChecked(True)
        self.ideal_filter.toggled.connect(self.filter_order.setDisabled)
        self.filter_order.setDisabled(True)
        form.addRow(self.ideal_filter)

        self.frequency_filter_button = QPushButton("Apply frequency filter")
        self.frequency_filter_button.clicked.connect(self.apply_frequency_filter)
        form.addRow(self.frequency_filter_button)

        self._update_frequency_filter_state()
        return group

    @staticmethod
    def _double_spin(
        minimum: float, maximum: float, value: float, decimals: int
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setValue(value)
        spin.setKeyboardTracking(False)
        return spin