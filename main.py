"""DSP Project: main window with one tab per signal type.

Each team writes a QWidget class in its own package:
    created_signal/gui.py -> class CreatedSignalTab(QWidget)
    image_signal/gui.py   -> class ImageSignalTab(QWidget)
    speech_signal/gui.py  -> class SpeechSignalTab(QWidget)
Until that file exists, main.py shows a placeholder, so nothing here has to change.
"""
import importlib
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QLabel, QMainWindow, QTabWidget,
                             QVBoxLayout, QWidget)

TABS = [
    ("Created Signal", "created_signal.gui", "CreatedSignalTab"),
    ("Image", "image_signal.gui", "ImageSignalTab"),
    ("Speech", "speech_signal.gui", "SpeechSignalTab"),
]


def _placeholder(text):
    w = QWidget()
    lay = QVBoxLayout(w)
    label = QLabel(text)
    label.setAlignment(Qt.AlignCenter)
    lay.addWidget(label)
    return w


def load_tab(module_name, class_name):
    """Import a team's tab class; return a placeholder if it is missing or broken."""
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)()
    except ModuleNotFoundError as e:
        if e.name == module_name:                    # the team's file does not exist yet
            return _placeholder(f"{class_name} is not implemented yet\n({module_name})")
        raise                                        # a real missing dependency: show it
    except AttributeError:
        return _placeholder(f"{module_name} has no class {class_name}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DSP Project")
        self.resize(1200, 800)
        self.tabs = QTabWidget()
        for title, module_name, class_name in TABS:
            self.tabs.addTab(load_tab(module_name, class_name), title)
        self.setCentralWidget(self.tabs)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
