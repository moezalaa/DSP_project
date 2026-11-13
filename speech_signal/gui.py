"""Speech tab (PyQt5): load / record audio, play it, filter it, add echo,
reduce noise, and look at the waveform, magnitude, phase and spectrogram.

Used by main.py as  ("Speech", "speech_signal.gui", "SpeechSignalTab").

Rule from CONTRACT.md: this file does NO signal processing. Every calculation
is a call into speech_signal.processing; the GUI only collects parameters,
catches ValueError (-> message box) and draws the returned arrays.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.gui_widgets import PlotCanvas, show_error
from core.utils import to_db
from speech_signal import processing as sp

DEFAULT_FS = 44100


class _RecordThread(QThread):
    """Runs the blocking sp.record_audio() away from the GUI thread."""

    done = pyqtSignal(object, int)      # (audio array, fs)
    failed = pyqtSignal(str)

    def __init__(self, duration: float, fs: int, parent=None):
        super().__init__(parent)
        self.duration = duration
        self.fs = fs

    def run(self):
        try:
            x = sp.record_audio(self.duration, self.fs)
        except Exception as e:           # ValueError from core, or anything unexpected
            self.failed.emit(str(e))
            return
        self.done.emit(x, self.fs)


class SpeechSignalTab(QWidget):
    """Speech tab loaded by main.py."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # State: `original` never changes after loading; `current` is what the
        # filters/echo/noise buttons build on (they can be chained).
        self.original: Optional[np.ndarray] = None
        self.current: Optional[np.ndarray] = None
        self.fs: Optional[int] = None
        self.history: list[str] = []
        self._orig_spectrum = None               # cache of compute_fft(original)
        self._record_thread: Optional[_RecordThread] = None

        self._build_ui()
        self._set_audio_controls_enabled(False)
        self._set_status("Load a WAV file or record from the microphone to get started.")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        controls = QWidget()
        lay = QVBoxLayout(controls)
        lay.setContentsMargins(4, 4, 8, 4)
        lay.setSpacing(10)
        lay.addWidget(self._build_input_group())
        lay.addWidget(self._build_playback_group())
        self.process_widgets = [
            self._build_filter_group(),
            self._build_echo_group(),
            self._build_noise_group(),
        ]
        for w in self.process_widgets:
            lay.addWidget(w)
        lay.addWidget(self._build_view_group())
        lay.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(controls)
        scroll.setMinimumWidth(400)
        scroll.setMaximumWidth(460)
        splitter.addWidget(scroll)

        # ---- right side: status bar + plots ----
        right = QWidget()
        rlay = QVBoxLayout(right)
        rlay.setContentsMargins(4, 4, 4, 4)

        header = QHBoxLayout()
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #666;")
        header.addWidget(self.status_label, 1)

        self.compare_check = QCheckBox("Compare with original")
        self.compare_check.setChecked(True)
        self.compare_check.toggled.connect(self._refresh_plots)
        header.addWidget(self.compare_check)

        self.reset_button = QPushButton("Reset")
        self.reset_button.setToolTip("Go back to the original audio")
        self.reset_button.clicked.connect(self.reset_audio)
        header.addWidget(self.reset_button)

        self.save_button = QPushButton("Save processed…")
        self.save_button.clicked.connect(self.save_audio_file)
        header.addWidget(self.save_button)
        rlay.addLayout(header)

        self.plot_tabs = QTabWidget()
        self.canvas = PlotCanvas(rows=3, cols=1)          # waveform / magnitude / phase
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.spec_canvas = PlotCanvas(rows=1, cols=1)     # spectrogram (drawn manually)
        self.plot_tabs.addTab(self.canvas, "Time && Spectrum")
        self.plot_tabs.addTab(self.spec_canvas, "Spectrogram")
        self.plot_tabs.currentChanged.connect(self._refresh_plots)
        rlay.addWidget(self.plot_tabs, 1)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 780])

    @staticmethod
    def _double_spin(lo, hi, value, decimals=2, step=None) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(decimals)
        s.setValue(value)
        if step is not None:
            s.setSingleStep(step)
        return s

    def _build_input_group(self) -> QGroupBox:
        g = QGroupBox("Input")
        form = QFormLayout(g)

        self.load_button = QPushButton("Load audio file…")
        self.load_button.clicked.connect(self.load_audio_file)
        form.addRow(self.load_button)

        self.rec_duration = self._double_spin(0.5, 600, 5.0, 1, 0.5)
        form.addRow("Record duration (s)", self.rec_duration)
        self.rec_fs = QComboBox()
        for f in (8000, 16000, 22050, 44100, 48000):
            self.rec_fs.addItem(str(f), f)
        self.rec_fs.setCurrentIndex(3)
        form.addRow("Record rate (Hz)", self.rec_fs)

        self.record_button = QPushButton("● Record")
        self.record_button.clicked.connect(self.start_recording)
        form.addRow(self.record_button)
        return g

    def _build_playback_group(self) -> QGroupBox:
        g = QGroupBox("Playback")
        row = QHBoxLayout(g)
        self.play_orig_button = QPushButton("▶ Original")
        self.play_orig_button.clicked.connect(lambda: self._play(self.original))
        self.play_proc_button = QPushButton("▶ Processed")
        self.play_proc_button.clicked.connect(lambda: self._play(self.current))
        self.stop_button = QPushButton("■ Stop")
        self.stop_button.clicked.connect(self._stop)
        for b in (self.play_orig_button, self.play_proc_button, self.stop_button):
            row.addWidget(b)
        return g

    def _build_filter_group(self) -> QGroupBox:
        g = QGroupBox("Frequency filter")
        form = QFormLayout(g)

        self.filter_type = QComboBox()
        self.filter_type.addItems(["lowpass", "highpass", "bandpass", "notch"])
        self.filter_type.currentTextChanged.connect(self._update_filter_state)
        form.addRow("Type", self.filter_type)

        self.cutoff_label = QLabel("Cutoff (Hz)")
        self.cutoff = self._double_spin(1, 192000, 1000, 1, 50)
        form.addRow(self.cutoff_label, self.cutoff)

        self.cutoff_high_label = QLabel("High cutoff (Hz)")
        self.cutoff_high = self._double_spin(1, 192000, 3400, 1, 50)
        form.addRow(self.cutoff_high_label, self.cutoff_high)

        self.notch_width_label = QLabel("Notch width (Hz)")
        self.notch_width = self._double_spin(0.1, 10000, 50, 1, 5)
        form.addRow(self.notch_width_label, self.notch_width)

        self.butterworth_check = QCheckBox("Butterworth (smooth)")
        self.butterworth_check.setToolTip("Unchecked = brick-wall filter (can 'ring' near the cutoff)")
        self.butterworth_check.toggled.connect(self._update_filter_state)
        form.addRow(self.butterworth_check)
        self.order_label = QLabel("Order")
        self.order = QSpinBox()
        self.order.setRange(1, 20)
        self.order.setValue(4)
        form.addRow(self.order_label, self.order)

        row = QHBoxLayout()
        phone = QPushButton("Telephone band")
        phone.setToolTip("Band-pass 300–3400 Hz: speech stays intelligible")
        phone.clicked.connect(self._telephone_preset)
        apply_btn = QPushButton("Apply filter")
        apply_btn.clicked.connect(self.apply_filter)
        row.addWidget(phone)
        row.addWidget(apply_btn)
        form.addRow(row)

        self._update_filter_state()
        return g

    def _build_echo_group(self) -> QGroupBox:
        g = QGroupBox("Echo (convolution)")
        form = QFormLayout(g)
        self.echo_delay = self._double_spin(0.01, 5.0, 0.25, 2, 0.05)
        form.addRow("Delay (s)", self.echo_delay)
        self.echo_decay = self._double_spin(0.0, 0.99, 0.5, 2, 0.05)
        form.addRow("Decay (0–0.99)", self.echo_decay)
        self.echo_count = QSpinBox()
        self.echo_count.setRange(1, 20)
        self.echo_count.setValue(2)
        form.addRow("Repeats", self.echo_count)
        btn = QPushButton("Apply echo")
        btn.setToolTip("The result is longer than the original (echo tail).")
        btn.clicked.connect(self.apply_echo)
        form.addRow(btn)
        return g

    def _build_noise_group(self) -> QGroupBox:
        g = QGroupBox("Noise reduction (spectral subtraction)")
        form = QFormLayout(g)
        self.alpha = self._double_spin(0.0, 10.0, 2.0, 2, 0.25)
        self.alpha.setToolTip("Over-subtraction: bigger removes more noise but adds artefacts")
        form.addRow("Alpha", self.alpha)
        self.beta = self._double_spin(0.0, 1.0, 0.05, 3, 0.01)
        self.beta.setToolTip("Spectral floor: keeps a little signal to avoid 'musical noise'")
        form.addRow("Beta", self.beta)

        self.noise_source = QComboBox()
        self.noise_source.addItems(["Automatic (quietest frames)", "First N seconds are noise only"])
        self.noise_source.currentIndexChanged.connect(self._update_noise_state)
        form.addRow("Noise estimate", self.noise_source)
        self.noise_dur_label = QLabel("Noise length (s)")
        self.noise_dur = self._double_spin(0.05, 60, 0.5, 2, 0.1)
        form.addRow(self.noise_dur_label, self.noise_dur)

        btn = QPushButton("Reduce noise")
        btn.clicked.connect(self.apply_noise_reduction)
        form.addRow(btn)
        self._update_noise_state()
        return g

    def _build_view_group(self) -> QGroupBox:
        g = QGroupBox("View")
        form = QFormLayout(g)
        self.max_freq = self._double_spin(100, 192000, 8000, 0, 500)
        self.max_freq.valueChanged.connect(self._refresh_plots)
        form.addRow("Max frequency shown (Hz)", self.max_freq)
        self.db_check = QCheckBox("Magnitude in dB")
        self.db_check.toggled.connect(self._refresh_plots)
        form.addRow(self.db_check)
        self.win_combo = QComboBox()
        for n in (256, 512, 1024, 2048):
            self.win_combo.addItem(str(n), n)
        self.win_combo.setCurrentIndex(2)
        self.win_combo.currentIndexChanged.connect(self._refresh_plots)
        form.addRow("Spectrogram window", self.win_combo)
        return g

    # ------------------------------------------------------------------
    # Small UI state helpers
    # ------------------------------------------------------------------
    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def _set_audio_controls_enabled(self, enabled: bool) -> None:
        for w in self.process_widgets:
            w.setEnabled(enabled)
        for w in (self.compare_check, self.reset_button, self.save_button,
                  self.play_orig_button, self.play_proc_button, self.stop_button):
            w.setEnabled(enabled)

    def _update_filter_state(self, *_args) -> None:
        kind = self.filter_type.currentText()
        band = kind == "bandpass"
        self.cutoff_label.setText("Low cutoff (Hz)" if band else
                                  "Center (Hz)" if kind == "notch" else "Cutoff (Hz)")
        for w in (self.cutoff_high_label, self.cutoff_high):
            w.setVisible(band)
        for w in (self.notch_width_label, self.notch_width):
            w.setVisible(kind == "notch")
        smooth = self.butterworth_check.isChecked()
        self.order.setEnabled(smooth)
        self.order_label.setEnabled(smooth)

    def _update_noise_state(self, *_args) -> None:
        use_clip = self.noise_source.currentIndex() == 1
        self.noise_dur.setEnabled(use_clip)
        self.noise_dur_label.setEnabled(use_clip)

    def _telephone_preset(self) -> None:
        self.filter_type.setCurrentText("bandpass")
        self.cutoff.setValue(300)
        self.cutoff_high.setValue(3400)

    def _require_audio(self) -> bool:
        if self.current is None or self.fs is None:
            show_error(self, "Load or record some audio first.")
            return False
        return True

    # ------------------------------------------------------------------
    # Input: load / record / save
    # ------------------------------------------------------------------
    def _install_audio(self, x: np.ndarray, fs: int, name: str) -> None:
        self._stop()
        self.original = np.asarray(x, dtype=float)
        self.current = self.original.copy()
        self.fs = int(fs)
        self.history = [name]
        self._orig_spectrum = None
        self.max_freq.setValue(min(8000.0, self.fs / 2))
        self._set_audio_controls_enabled(True)
        self._set_status(f"{name}: {len(x)} samples, {self.fs} Hz, {len(x) / self.fs:.2f} s")
        self._refresh_plots()

    def load_audio_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open audio file", "", "Audio files (*.wav *.flac *.ogg);;All files (*)")
        if path:
            self.load_path(path)

    def load_path(self, path: str) -> bool:
        """Load an audio file by path (also used by the tests). Returns True on success."""
        try:
            x, fs = sp.load_audio(path)
        except ValueError as e:
            show_error(self, e)
            return False
        self._install_audio(x, fs, path.replace("\\", "/").split("/")[-1])
        return True

    def start_recording(self) -> None:
        if self._record_thread is not None and self._record_thread.isRunning():
            return
        duration = self.rec_duration.value()
        fs = int(self.rec_fs.currentData())
        self._stop()
        self.record_button.setEnabled(False)
        self.load_button.setEnabled(False)
        self._set_status(f"Recording {duration:g} s… speak now.")
        self._record_thread = _RecordThread(duration, fs, self)
        self._record_thread.done.connect(self._on_recorded)
        self._record_thread.failed.connect(self._on_record_failed)
        self._record_thread.start()

    def _on_recorded(self, x, fs) -> None:
        self.record_button.setEnabled(True)
        self.load_button.setEnabled(True)
        self._install_audio(x, fs, "Recording")

    def _on_record_failed(self, message: str) -> None:
        self.record_button.setEnabled(True)
        self.load_button.setEnabled(True)
        self._set_status("Recording failed.")
        show_error(self, message)

    def save_audio_file(self) -> None:
        if not self._require_audio():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save processed audio", "processed.wav", "WAV file (*.wav)")
        if not path:
            return
        if not path.lower().endswith(".wav"):
            path += ".wav"
        try:
            sp.save_audio(path, self.current, self.fs)
        except ValueError as e:
            show_error(self, e)
            return
        self._set_status(f"Saved to {path}")

    # ------------------------------------------------------------------
    # Playback (sounddevice is imported lazily: a missing PortAudio must not
    # stop the whole app from starting)
    # ------------------------------------------------------------------
    def _play(self, x: Optional[np.ndarray]) -> None:
        if x is None or self.fs is None:
            return
        try:
            import sounddevice as sd
            sd.stop()
            sd.play(np.asarray(x, dtype="float32"), self.fs)
        except Exception as e:
            show_error(self, f"Cannot play audio: {e}")

    def _stop(self) -> None:
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Processing buttons: each one calls ONE function of speech_signal.processing
    # ------------------------------------------------------------------
    def _run(self, label: str, func: Callable[[np.ndarray, int], np.ndarray]) -> None:
        if not self._require_audio():
            return
        self._stop()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            y = func(self.current, self.fs)
        except ValueError as e:
            show_error(self, e)
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.current = np.asarray(y, dtype=float)
        self.history.append(label)
        self._set_status(" → ".join(self.history) + f"   ({len(self.current) / self.fs:.2f} s)")
        self._refresh_plots()

    def apply_filter(self) -> None:
        kind = self.filter_type.currentText()
        low = self.cutoff.value()
        if kind == "bandpass":
            high = self.cutoff_high.value()
            if low >= high:
                show_error(self, "For a band-pass, the low cutoff must be below the high cutoff.")
                return
            cutoff, desc = (low, high), f"{low:g}–{high:g} Hz"
        else:
            cutoff, desc = low, f"{low:g} Hz"
        order = self.order.value() if self.butterworth_check.isChecked() else None
        width = self.notch_width.value()
        label = f"{kind} {desc}" + (f" (Butterworth {order})" if order else "")
        self._run(label, lambda x, fs: sp.filter_audio(x, fs, kind, cutoff, order=order, width=width))

    def apply_echo(self) -> None:
        delay, decay, n = self.echo_delay.value(), self.echo_decay.value(), self.echo_count.value()
        self._run(f"echo {delay:g}s ×{n}",
                  lambda x, fs: sp.add_echo(x, fs, delay, decay, n_echoes=n))

    def apply_noise_reduction(self) -> None:
        params = {"alpha": self.alpha.value(), "beta": self.beta.value()}
        if self.noise_source.currentIndex() == 1:
            params["noise_duration"] = self.noise_dur.value()
        self._run("noise reduction", lambda x, fs: sp.reduce_noise(x, fs, **params))

    def reset_audio(self) -> None:
        if self.original is None:
            return
        self._stop()
        self.current = self.original.copy()
        self.history = self.history[:1]
        self._set_status(f"{self.history[0]}: back to the original.")
        self._refresh_plots()

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    @property
    def _is_modified(self) -> bool:
        return len(self.history) > 1

    def _refresh_plots(self, *_args) -> None:
        if self.current is None or self.fs is None:
            self.canvas.clear()
            return
        try:
            if self.plot_tabs.currentIndex() == 0:
                self._draw_time_and_spectrum()
            else:
                self._draw_spectrogram()
        except ValueError as e:
            show_error(self, e)

    def _draw_time_and_spectrum(self) -> None:
        x, fs = self.current, self.fs
        compare = self.compare_check.isChecked() and self._is_modified
        fmax = min(self.max_freq.value(), fs / 2)

        t = np.arange(len(x)) / fs
        t0 = np.arange(len(self.original)) / fs
        labels = ("Processed", "Original") if compare else ("", "")
        self.canvas.plot(t, x, index=0, title="Waveform", xlabel="Time (s)", ylabel="Amplitude",
                         overlay=(t0, self.original) if compare else None, labels=labels)

        f, mag, ph = sp.compute_fft(x, fs)
        if self._orig_spectrum is None and compare:
            self._orig_spectrum = sp.compute_fft(self.original, fs)
        as_db = self.db_check.isChecked()
        scale = (lambda m: to_db(m)) if as_db else (lambda m: m)
        overlay = (self._orig_spectrum[0], scale(self._orig_spectrum[1])) if compare else None
        self.canvas.plot(f, scale(mag), index=1, title="Magnitude spectrum", xlabel="Frequency (Hz)",
                         ylabel="dB" if as_db else "|X|", overlay=overlay, labels=labels)
        self.canvas.plot(f, ph, index=2, title="Phase spectrum", xlabel="Frequency (Hz)", ylabel="rad")
        for ax in self.canvas.axes[1:]:
            ax.set_xlim(0, fmax)
        self.canvas.draw_idle()

    def _draw_spectrogram(self) -> None:
        x, fs = self.current, self.fs
        win = int(self.win_combo.currentData())
        times, freqs, S = sp.compute_spectrogram(x, fs, window_size=win, hop=win // 2)
        keep = freqs <= min(self.max_freq.value(), fs / 2)
        fig = self.spec_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        mesh = ax.pcolormesh(times, freqs[keep], S[keep], shading="auto",
                             vmin=S.max() - 90, vmax=S.max(), cmap="magma")
        fig.colorbar(mesh, ax=ax, label="dB")
        ax.set_title(f"Spectrogram (Hamming {win}, hop {win // 2})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        self.spec_canvas.draw_idle()
