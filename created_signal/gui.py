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

        