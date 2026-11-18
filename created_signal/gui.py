"""Created Signal tab (PyQt5): build a signal, add noise, filter it, see
time / magnitude / phase, and compare our code against numpy / scipy.

main.py loads CreatedSignalTab automatically.
"""
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox,
                             QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPushButton,
                             QScrollArea, QTableWidget, QTableWidgetItem, QTabWidget,
                             QVBoxLayout, QWidget)

from core.fourier import fft
from core.gui_widgets import LabeledSlider, PlotCanvas, show_error
from core.utils import fftshift
from created_signal import compare, processing as P

FILTER_LABELS = {
    "none": "None",
    "moving_average": "Moving average (convolution)",
    "difference": "Difference [1, -1] (convolution)",
    "gaussian": "Gaussian smoothing (convolution)",
    "custom": "Custom kernel (convolution)",
    "first_order": "Difference equation: y[n] = a*y[n-1] + b*x[n]",
    "lowpass": "Low-pass (Fourier)",
    "highpass": "High-pass (Fourier)",
    "bandpass": "Band-pass (Fourier)",
    "notch": "Notch (Fourier)",
}

DEFAULT_COMPONENTS = [
    dict(on=True, kind="sine", amp=1.0, freq=5.0, phase=0.0),
    dict(on=True, kind="sine", amp=0.5, freq=40.0, phase=0.0),
    dict(on=False, kind="square", amp=0.5, freq=10.0, phase=0.0),
]


class ComponentBox(QGroupBox):
    """One waveform component: on/off, type, amplitude, frequency, phase (degrees)."""

    def __init__(self, index, on_change, parent=None):
        super().__init__(f"Component {index}", parent)
        self.setCheckable(True)
        self.kind = QComboBox()
        self.kind.addItems(P.WAVE_KINDS)
        self.amp = LabeledSlider("Amplitude", 0, 10, 1, step=0.1, decimals=2)
        self.freq = LabeledSlider("Frequency (Hz)", 0, 2000, 5, step=0.5, decimals=2)
        self.phase = LabeledSlider("Phase (deg)", -180, 180, 0, step=5, decimals=1)
        lay = QVBoxLayout(self)
        lay.addWidget(self.kind)
        for w in (self.amp, self.freq, self.phase):
            lay.addWidget(w)
            w.valueChanged.connect(lambda _v: on_change())
        self.kind.currentIndexChanged.connect(lambda _i: on_change())
        self.toggled.connect(lambda _b: on_change())

    def set_values(self, on, kind, amp, freq, phase):
        self.setChecked(on)
        self.kind.setCurrentText(kind)
        self.amp.setValue(amp)
        self.freq.setValue(freq)
        self.phase.setValue(phase)

    def get(self):
        """Component dict for processing (phase in radians)."""
        return {"kind": self.kind.currentText(), "amp": self.amp.value(),
                "freq": self.freq.value(), "phase": np.deg2rad(self.phase.value())}


class CreatedSignalTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.result = None
        self.comparison = None
        self._seed = 0
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(150)               # wait for the user to stop moving a slider
        self._timer.timeout.connect(self.update_plots)
        self._build_ui()
        self.reset()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        root = QHBoxLayout(self)

        # ---- left: controls (scrollable) ----
        panel = QWidget()
        left = QVBoxLayout(panel)

        g = QGroupBox("Sampling and display")
        gl = QVBoxLayout(g)
        self.fs = LabeledSlider("Sampling rate (Hz)", 100, 10000, 1000, step=100, decimals=0)
        self.duration = LabeledSlider("Duration (s)", 0.1, 5, 1, step=0.1, decimals=1)
        gl.addWidget(self.fs)
        gl.addWidget(self.duration)
        self.fs.valueChanged.connect(lambda _v: self._changed())
        self.duration.valueChanged.connect(lambda _v: self._changed())
        self.phase_thr = LabeledSlider("Phase threshold (%)", 0, 100, 5, step=1, decimals=0)
        self.phase_thr.valueChanged.connect(lambda _v: self._changed())
        gl.addWidget(self.phase_thr)
        left.addWidget(g)

        self.components = [ComponentBox(i + 1, self._changed) for i in range(3)]
        for c in self.components:
            left.addWidget(c)

        self.noise_box = QGroupBox("Gaussian noise")
        self.noise_box.setCheckable(True)
        nl = QHBoxLayout(self.noise_box)
        self.snr = LabeledSlider("SNR (dB)", -10, 40, 15, step=1, decimals=0)
        self.new_noise = QPushButton("New noise")
        nl.addWidget(self.snr, 1)
        nl.addWidget(self.new_noise)
        self.snr.valueChanged.connect(lambda _v: self._changed())
        self.noise_box.toggled.connect(lambda _b: self._changed())
        self.new_noise.clicked.connect(self._reseed)
        left.addWidget(self.noise_box)

        fg = QGroupBox("Filter")
        fl = QVBoxLayout(fg)
        self.filter_type = QComboBox()
        for key in P.FILTER_KINDS:
            self.filter_type.addItem(FILTER_LABELS[key], key)
        fl.addWidget(self.filter_type)
        self.size = LabeledSlider("Kernel size", 1, 101, 5, step=1, decimals=0)
        self.kernel_text = QLineEdit("1, 2, 1")
        self.kernel_row = QWidget()
        kl = QFormLayout(self.kernel_row)
        kl.setContentsMargins(0, 0, 0, 0)
        kl.addRow("Kernel values", self.kernel_text)
        self.cutoff = LabeledSlider("Cutoff / center (Hz)", 0.5, 5000, 20, step=0.5, decimals=1)
        self.cutoff_high = LabeledSlider("High cutoff (Hz)", 0.5, 5000, 60, step=0.5, decimals=1)
        self.order = LabeledSlider("Order (0 = ideal)", 0, 10, 0, step=1, decimals=0)
        self.width = LabeledSlider("Notch width (Hz)", 0.5, 50, 4, step=0.5, decimals=1)
        self.de_a = LabeledSlider("a (feedback)", -1.2, 1.2, 0.9, step=0.01, decimals=2)
        self.de_b = LabeledSlider("b (input gain)", -2, 2, 0.1, step=0.05, decimals=2)
        for w in (self.size, self.kernel_row, self.de_a, self.de_b, self.cutoff,
                  self.cutoff_high, self.order, self.width):
            fl.addWidget(w)
        for w in (self.size, self.de_a, self.de_b, self.cutoff, self.cutoff_high, self.order, self.width):
            w.valueChanged.connect(lambda _v: self._changed())
        self.kernel_text.editingFinished.connect(self._changed)
        self.filter_type.currentIndexChanged.connect(self._filter_changed)
        left.addWidget(fg)

        row = QHBoxLayout()
        self.auto = QCheckBox("Auto update")
        self.auto.setChecked(True)
        self.btn_apply = QPushButton("Apply")
        self.btn_reset = QPushButton("Reset")
        row.addWidget(self.auto)
        row.addWidget(self.btn_apply)
        row.addWidget(self.btn_reset)
        left.addLayout(row)
        row2 = QHBoxLayout()
        self.btn_csv = QPushButton("Save CSV")
        self.btn_png = QPushButton("Save plots (PNG)")
        row2.addWidget(self.btn_csv)
        row2.addWidget(self.btn_png)
        left.addLayout(row2)
        self.btn_apply.clicked.connect(self.update_plots)
        self.btn_reset.clicked.connect(self.reset)
        self.btn_csv.clicked.connect(self.save_csv)
        self.btn_png.clicked.connect(self.save_png)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        left.addWidget(self.status)
        left.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(panel)
        scroll.setFixedWidth(400)
        root.addWidget(scroll)

        # ---- right: plots + comparison ----
        self.right = QTabWidget()
        self.canvas = PlotCanvas(rows=3)
        self.right.addTab(self.canvas, "Signals")

        cmp_page = QWidget()
        cl = QVBoxLayout(cmp_page)
        self.cmp_summary = QLabel("")
        self.cmp_summary.setWordWrap(True)
        cl.addWidget(self.cmp_summary)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["Operation", "Reference", "Ours ms", "Ref ms",
             "Max |err|", "RMSE", "OK"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setFixedHeight(340)
        cl.addWidget(self.table)
        self.err_canvas = PlotCanvas(rows=3)
        cl.addWidget(self.err_canvas, 1)
        self.right.addTab(cmp_page, "Accuracy: ours vs NumPy/SciPy")
        self.right.addTab(self._build_dtft_page(), "DTFT (loops)")
        self.right.addTab(self._build_conv_page(), "Convolution vs FFT")
        self.right.currentChanged.connect(lambda _i: self._refresh_active_tab())
        root.addWidget(self.right, 1)

    def _build_dtft_page(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        row = QHBoxLayout()
        self.dtft_n = LabeledSlider("Samples used (N)", 8, 1000, 256, step=8, decimals=0)
        self.dtft_k = LabeledSlider("Frequency points", 16, 1000, 256, step=8, decimals=0)
        self.btn_dtft = QPushButton("Run DTFT")
        row.addWidget(self.dtft_n, 1)
        row.addWidget(self.dtft_k, 1)
        row.addWidget(self.btn_dtft)
        lay.addLayout(row)
        self.dtft_info = QLabel("")
        self.dtft_info.setWordWrap(True)
        lay.addWidget(self.dtft_info)
        self.dtft_canvas = PlotCanvas(rows=2)
        lay.addWidget(self.dtft_canvas, 1)
        self.btn_dtft.clicked.connect(self.run_dtft)
        return page

    def _build_conv_page(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        self.conv_info = QLabel("")
        self.conv_info.setWordWrap(True)
        lay.addWidget(self.conv_info)
        self.conv_canvas = PlotCanvas(rows=2)
        lay.addWidget(self.conv_canvas, 1)
        return page

    # ------------------------------------------------------------ helpers
    def _filter_changed(self):
        kind = self.filter_type.currentData()
        show = {
            "size": kind in ("moving_average", "gaussian"),
            "kernel": kind == "custom",
            "de": kind == "first_order",
            "cutoff": kind in P.FREQ_KINDS,
            "cutoff_high": kind == "bandpass",
            "order": kind in ("lowpass", "highpass", "bandpass"),
            "width": kind == "notch",
        }
        self.size.setVisible(show["size"])
        self.kernel_row.setVisible(show["kernel"])
        self.de_a.setVisible(show["de"])
        self.de_b.setVisible(show["de"])
        self.cutoff.setVisible(show["cutoff"])
        self.cutoff_high.setVisible(show["cutoff_high"])
        self.order.setVisible(show["order"])
        self.width.setVisible(show["width"])
        self.cutoff.label.setText("Center (Hz)" if kind == "notch" else
                                  "Low cutoff (Hz)" if kind == "bandpass" else "Cutoff (Hz)")
        self._changed()

    def _changed(self):
        if self.auto.isChecked():
            self._timer.start()

    def _reseed(self):
        self._seed += 1
        self.update_plots()

    def _parse_kernel(self):
        text = self.kernel_text.text().replace(";", ",").replace(" ", ",")
        try:
            vals = [float(v) for v in text.split(",") if v.strip()]
        except ValueError:
            raise ValueError("Kernel values must be numbers separated by commas")
        if not vals:
            raise ValueError("Custom kernel is empty")
        return vals

    def filter_spec(self):
        """Filter chosen in the GUI, as a dict for processing.apply_filter_spec."""
        kind = self.filter_type.currentData()
        spec = {"type": kind, "size": int(self.size.value()),
                "cutoff": self.cutoff.value(), "cutoff_high": self.cutoff_high.value(),
                "order": int(self.order.value()) or None, "width": self.width.value(),
                "a": self.de_a.value(), "b": self.de_b.value()}
        if kind == "custom":
            spec["kernel"] = self._parse_kernel()
        return spec

    def _comparison_specs(self):
        """One time-domain and one frequency-domain filter for the comparison table."""
        spec = self.filter_spec()
        conv = spec if spec["type"] in P.CONV_KINDS else {**spec, "type": "moving_average"}
        freq = spec if spec["type"] in P.FREQ_KINDS else {**spec, "type": "lowpass"}
        de = {"type": "first_order", "a": spec["a"], "b": spec["b"]}
        return conv, freq, de

    def params(self):
        comps = [c.get() for c in self.components if c.isChecked()]
        return {"fs": self.fs.value(), "duration": self.duration.value(), "components": comps,
                "noise_snr": self.snr.value() if self.noise_box.isChecked() else None,
                "seed": self._seed, "filter": self.filter_spec()}

    # ------------------------------------------------------------- actions
    def reset(self):
        was_auto = self.auto.isChecked()
        self.auto.setChecked(False)                       # no recompute storm while resetting
        self.fs.setValue(1000)
        self.duration.setValue(1.0)
        self.phase_thr.setValue(5)
        for box, d in zip(self.components, DEFAULT_COMPONENTS):
            box.set_values(d["on"], d["kind"], d["amp"], d["freq"], d["phase"])
        self.noise_box.setChecked(False)
        self.snr.setValue(15)
        self.filter_type.setCurrentIndex(P.FILTER_KINDS.index("none"))
        self.size.setValue(5)
        self.cutoff.setValue(20)
        self.cutoff_high.setValue(60)
        self.order.setValue(0)
        self.width.setValue(4)
        self.de_a.setValue(0.9)
        self.de_b.setValue(0.1)
        self.dtft_n.setValue(256)
        self.dtft_k.setValue(256)
        self.kernel_text.setText("1, 2, 1")
        self._filter_changed()
        self.auto.setChecked(was_auto)
        self.update_plots()

    def update_plots(self):
        self._timer.stop()
        try:
            params = self.params()
            self.result = P.run_pipeline(params)
        except ValueError as e:
            self._set_status(str(e), error=True)
            return
        r, fs = self.result, params["fs"]
        warnings = P.nyquist_warnings(params["components"], fs)
        f = params["filter"]
        if f["type"] == "first_order" and not P.is_stable(f["a"]):
            warnings.append(f"Difference equation is UNSTABLE: |a| = {abs(f['a']):g} must be < 1")
        self._set_status("  |  ".join(warnings) if warnings else "OK", error=bool(warnings))

        fmax = self._spectrum_limit(params)
        self.canvas.plot(r["t"], r["input"], index=0, title="Time domain", xlabel="Time (s)",
                         ylabel="Amplitude", overlay=(r["t"], r["output"]),
                         labels=("Input", "Output"))
        self.canvas.plot(r["freqs"], r["mag_in"], index=1, title="Magnitude spectrum",
                         xlabel="Frequency (Hz)", ylabel="Amplitude",
                         overlay=(r["freqs"], r["mag_out"]), labels=("Input", "Output"))
        frac = self.phase_thr.value() / 100.0
        ph_in = self._visible_phase(r["ph_in"], r["mag_in"], frac)
        ph_out = self._visible_phase(r["ph_out"], r["mag_out"], frac)
        self.canvas.plot(r["freqs"], ph_in, index=2,
                         title=f"Phase spectrum (bins above {self.phase_thr.value():g}% of the peak)",
                         xlabel="Frequency (Hz)", ylabel="Phase (rad)",
                         overlay=(r["freqs"], ph_out), labels=("Input", "Output"), marker="o")
        self.canvas.axes[1].set_xlim(0, fmax)
        self.canvas.axes[2].set_xlim(0, fmax)
        self.canvas.axes[2].set_ylim(-np.pi - 0.2, np.pi + 0.2)
        self.canvas.draw_idle()
        self._refresh_active_tab()

    @staticmethod
    def _visible_phase(ph, mag, fraction=0.05):
        """Hide the phase of bins that carry almost no energy (it is just noise).

        The strongest bin is always kept, so the phase plot is never empty.
        """
        keep = mag >= fraction * mag.max()
        keep[np.argmax(mag)] = True
        return np.where(keep, ph, np.nan)

    def _spectrum_limit(self, params):
        """Frequency range worth showing: a bit above the highest tone / cutoff."""
        nyq = params["fs"] / 2
        top = max([c["freq"] for c in params["components"]] + [1.0])
        f = params["filter"]
        if f["type"] in P.FREQ_KINDS:
            top = max(top, f["cutoff"], f["cutoff_high"] if f["type"] == "bandpass" else 0)
        return float(min(nyq, max(top * 3, 20)))

    def _set_status(self, text, error=False):
        self.status.setText(text)
        self.status.setStyleSheet("color: #c0392b;" if error else "color: #2e8b57;")

    # ---------------------------------------------------------- comparison
    def _refresh_active_tab(self):
        """Recompute only the extra tab that is visible (they are slower than the main plots)."""
        if self.result is None:
            return
        idx = self.right.currentIndex()
        if idx == 1:
            self.run_comparison()
        elif idx == 2:
            self.run_dtft()
        elif idx == 3:
            self.run_convolution()

    _refresh_comparison = _refresh_active_tab        # old name, kept for compatibility

    def run_comparison(self):
        params = self.params()
        try:
            conv, freq, de = self._comparison_specs()
            self.comparison = compare.compare_all(
                params["components"], params["fs"], params["duration"],
                self.result["input"], conv, freq, de)
        except ValueError as e:
            self._set_status(str(e), error=True)
            return
        self._fill_table(self.comparison)
        self._plot_errors(self.comparison)

    def _fill_table(self, cmp):
        rows = cmp["rows"]
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            cells = [r["name"], r["reference"], f'{r["ours_ms"]:.3f}', f'{r["ref_ms"]:.3f}',
                     f'{r["max_abs"]:.2e}', f'{r["rmse"]:.2e}', "✓" if r["ok"] else "✗"]
            for j, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if j >= 2:
                    item.setTextAlignment(Qt.AlignCenter)
                if j == 6:
                    item.setForeground(QColor("#2e8b57" if r["ok"] else "#c0392b"))
                self.table.setItem(i, j, item)
        worst = max(r["max_abs"] for r in rows)
        if cmp["all_ok"]:
            msg = (f"All {len(rows)} operations match NumPy/SciPy "
                   f"(worst max error {worst:.2e}).")
        else:
            bad = [r["name"] for r in rows if not r["ok"]]
            msg = "Mismatch in: " + ", ".join(bad)
        self.cmp_summary.setText(msg)
        self.cmp_summary.setStyleSheet("color: #2e8b57;" if cmp["all_ok"] else "color: #c0392b;")

    def _plot_errors(self, cmp):
        floor = 1e-18                                           # so log scale can show exact zeros
        e = cmp["errors"]
        t = self.result["t"]
        spec = [("Convolution filter error (ours - SciPy)", t, e["conv"], "Time (s)"),
                ("Frequency filter error (ours - NumPy)", t, e["freq"], "Time (s)"),
                ("Magnitude spectrum error (ours - NumPy)", e["freqs"], e["magnitude"],
                 "Frequency (Hz)")]
        for i, (title, x, err, xl) in enumerate(spec):
            self.err_canvas.plot(x, np.maximum(np.abs(err), floor), index=i, title=title,
                                 xlabel=xl, ylabel="|error|")
            self.err_canvas.axes[i].set_yscale("log")
            self.err_canvas.axes[i].set_ylim(1e-18, 1e-3)
        self.err_canvas.draw_idle()

    # ---------------------------------------------------------------- DTFT
    def run_dtft(self):
        """DTFT from the defining equation (explicit loops) of the first N samples."""
        if self.result is None:
            return
        n, k = int(self.dtft_n.value()), int(self.dtft_k.value())
        try:
            w, X, mag, ph = P.dtft_spectrum(self.result["input"], n, k)
        except ValueError as e:
            self._set_status(str(e), error=True)
            return
        fs = self.fs.value()
        n_used = min(n, len(self.result["input"]))
        c = self.dtft_canvas
        c.plot(w, mag, index=0, title="DTFT magnitude  |X(w)|", xlabel="w (rad/sample)",
               ylabel="|X(w)|", labels=("DTFT (loops)", ""))
        # the DFT (FFT) is the DTFT sampled at w_k = 2*pi*k/N: draw those samples on top
        xs = self.result["input"][:n_used]
        dft = fftshift(fft(xs))
        w_dft = 2 * np.pi * (np.arange(n_used) - n_used // 2) / n_used
        c.axes[0].plot(w_dft, np.abs(dft), "o", markersize=3, color="#e67e22",
                       label="DFT samples (FFT)")
        c.axes[0].legend(loc="upper right")
        frac = self.phase_thr.value() / 100.0
        c.plot(w, self._visible_phase(ph, mag, frac), index=1,
               title=f"DTFT phase (bins above {self.phase_thr.value():g}% of the peak)",
               xlabel="w (rad/sample)", ylabel="Phase (rad)", marker="o")
        ticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
        names = ["-pi", "-pi/2", "0", "pi/2", "pi"]
        for ax in c.axes:
            ax.set_xticks(ticks)
            ax.set_xticklabels(names)
        c.axes[1].set_ylim(-np.pi - 0.2, np.pi + 0.2)
        c.draw_idle()
        self.dtft_info.setText(
            f"X(w) = sum over n of x[n]*exp(-j*w*n), with N = {n_used} samples and {k} frequency "
            f"points. w = 2*pi*f/fs, so w = pi corresponds to the Nyquist frequency "
            f"({fs / 2:g} Hz) and a 5 Hz tone sits at w = {2 * np.pi * 5 / fs:.4f}.")

    # --------------------------------------------------- convolution vs FFT
    def run_convolution(self):
        """Linear convolution (time) vs multiplication in frequency, with / without zero-padding."""
        if self.result is None:
            return
        x = self.result["input"]
        m = max(1, int(self.size.value()))
        h = P.make_kernel("moving_average", m)
        r = P.convolution_vs_frequency(x, h)
        c = self.conv_canvas
        idx_full = np.arange(len(r["direct"]))
        c.plot(idx_full, r["direct"], index=0,
               title=f"Moving average (M = {m}): time-domain convolution vs frequency domain",
               xlabel="Sample n", ylabel="y[n]",
               overlay=(idx_full, r["padded"]),
               labels=("Convolution (time domain)", "FFT x FFT, zero-padded"))
        floor = 1e-18
        e_pad = np.maximum(np.abs(r["err_padded"]), floor)
        e_circ = np.maximum(np.abs(r["err_circular"]), floor)
        c.plot(np.arange(len(e_pad)), e_pad, index=1,
               title="|error| against the true linear convolution", xlabel="Sample n",
               ylabel="|error|", overlay=(np.arange(len(e_circ)), e_circ),
               labels=("Zero-padded to N+M-1", "No zero-padding (circular)"))
        c.axes[1].set_yscale("log")
        c.axes[1].set_ylim(1e-18, 10)
        c.draw_idle()
        self.conv_info.setText(
            f"N = {len(x)}, M = {m}.  Zero-padded to N+M-1 = {len(r['direct'])}: max error "
            f"{np.max(np.abs(r['err_padded'])):.2e} (same as linear convolution).  Without "
            f"zero-padding the product of FFTs is a CIRCULAR convolution: max error "
            f"{np.max(np.abs(r['err_circular'])):.2e}, concentrated in the first M-1 = {m - 1} "
            f"samples where the tail wraps around.")

    # --------------------------------------------------------------- export
    def export_csv(self, path):
        """Write time, input, output (and the clean signal) to a CSV file."""
        if self.result is None:
            raise ValueError("Nothing to save yet")
        r = self.result
        data = np.column_stack([r["t"], r["clean"], r["input"], r["output"]])
        np.savetxt(path, data, delimiter=",", header="time_s,clean,input,output", comments="")

    def save_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save signals", "signal.csv", "CSV (*.csv)")
        if path:
            try:
                self.export_csv(path)
            except (ValueError, OSError) as e:
                show_error(self, e)

    def save_png(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save plots", "plots.png", "PNG (*.png)")
        if path:
            try:
                canvas = self.err_canvas if self.right.currentIndex() == 1 else self.canvas
                canvas.figure.savefig(path, dpi=150)
            except OSError as e:
                show_error(self, e)
