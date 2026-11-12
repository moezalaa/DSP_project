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

