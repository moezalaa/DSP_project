from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QPushButton, QFileDialog, QGroupBox, QScrollArea, QFrame,
    QComboBox, QSpinBox, QDoubleSpinBox
)

from .denoising import (add_periodic_noise, analyze_noisy_image, remove_periodic_noise)
from core.gui_widgets import ImageCanvas, show_error
from .processing import load_image, spatial_filter, image_fft, frequency_filter
from scripts.compare_image import calculate_metrics


PANEL_WIDTH = 400


class ImageSignalTab(QWidget):

    def __init__(self):
        super().__init__()

        self.original_image = None
        self.processed_image = None
        self.noisy_image = None

        self.init_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _make_group(self, title):
        """Create a QGroupBox with a QFormLayout and return both."""
        box = QGroupBox(title)
        form = QFormLayout(box)
        form.setLabelAlignment(form.labelAlignment())
        return box, form

    def init_ui(self):

        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ---------------- Left: control panel ----------------
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(6, 6, 6, 6)

        # --- Input ---
        input_box = QGroupBox("Input")
        input_layout = QVBoxLayout(input_box)

        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image_file)
        input_layout.addWidget(self.load_button)

        self.fft_button = QPushButton("Analyze FFT")
        self.fft_button.clicked.connect(self.analyze_fft)
        input_layout.addWidget(self.fft_button)

        panel_layout.addWidget(input_box)

        # --- Spatial filter ---
        spatial_box, spatial_form = self._make_group("Spatial filter")

        self.filter_combo = QComboBox()
        self.filter_combo.addItem("Box", "box")
        self.filter_combo.addItem("Gaussian", "gaussian")
        self.filter_combo.addItem("Sharpen", "sharpen")
        self.filter_combo.addItem("Sobel X", "sobel_x")
        self.filter_combo.addItem("Sobel Y", "sobel_y")
        self.filter_combo.addItem("Sobel", "sobel")
        self.filter_combo.addItem("Prewitt", "prewitt")
        self.filter_combo.addItem("Laplacian", "laplacian")
        spatial_form.addRow("Filter", self.filter_combo)

        self.kernel_size = QComboBox()
        self.kernel_size.addItems(["3", "5", "7", "9"])
        spatial_form.addRow("Kernel size", self.kernel_size)

        self.sigma = QDoubleSpinBox()
        self.sigma.setRange(0.1, 10.0)
        self.sigma.setSingleStep(0.1)
        self.sigma.setValue(1.0)
        spatial_form.addRow("Sigma", self.sigma)

        self.apply_button = QPushButton("Apply Filter")
        self.apply_button.clicked.connect(self.apply_spatial_filter)
        spatial_form.addRow(self.apply_button)

        panel_layout.addWidget(spatial_box)

        # --- Frequency filter ---
        freq_box, freq_form = self._make_group("Frequency filter")

        self.freq_type = QComboBox()
        self.freq_type.addItem("Low Pass", "lowpass")
        self.freq_type.addItem("High Pass", "highpass")
        freq_form.addRow("Type", self.freq_type)

        self.freq_kind = QComboBox()
        self.freq_kind.addItems(["ideal", "gaussian", "butterworth"])
        freq_form.addRow("Filter type", self.freq_kind)

        self.cutoff = QDoubleSpinBox()
        self.cutoff.setRange(0.1, 500)
        self.cutoff.setValue(30)
        self.cutoff.setSingleStep(1)
        freq_form.addRow("Cutoff", self.cutoff)

        self.order = QSpinBox()
        self.order.setRange(1, 10)
        self.order.setValue(2)
        freq_form.addRow("Order", self.order)

        self.freq_button = QPushButton("Apply Frequency Filter")
        self.freq_button.clicked.connect(self.apply_frequency_filter)
        freq_form.addRow(self.freq_button)

        # Order only matters for Butterworth (same greyed-out look as the other tabs)
        self.freq_kind.currentTextChanged.connect(self._update_freq_controls)
        self._update_freq_controls(self.freq_kind.currentText())

        panel_layout.addWidget(freq_box)

        # --- Periodic noise ---
        noise_box, noise_form = self._make_group("Periodic noise")

        self.noise_amplitude = QDoubleSpinBox()
        self.noise_amplitude.setRange(0, 255)
        self.noise_amplitude.setValue(30)
        noise_form.addRow("Amplitude", self.noise_amplitude)

        self.noise_fx = QSpinBox()
        self.noise_fx.setRange(1, 100)
        self.noise_fx.setValue(8)
        noise_form.addRow("Frequency X", self.noise_fx)

        self.noise_fy = QSpinBox()
        self.noise_fy.setRange(1, 100)
        self.noise_fy.setValue(6)
        noise_form.addRow("Frequency Y", self.noise_fy)

        self.add_noise_button = QPushButton("Add Periodic Noise")
        self.add_noise_button.clicked.connect(self.add_noise)
        noise_form.addRow(self.add_noise_button)

        panel_layout.addWidget(noise_box)

        # --- Noise removal ---
        remove_box, remove_form = self._make_group("Noise removal")

        self.notch_radius = QSpinBox()
        self.notch_radius.setRange(1, 20)
        self.notch_radius.setValue(3)
        remove_form.addRow("Notch radius", self.notch_radius)

        self.remove_noise_button = QPushButton("Remove Periodic Noise")
        self.remove_noise_button.clicked.connect(self.remove_noise)
        remove_form.addRow(self.remove_noise_button)

        panel_layout.addWidget(remove_box)

        panel_layout.addStretch(1)

        # --- Bottom actions: Reset / Save ---
        actions_layout = QHBoxLayout()

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_image)
        actions_layout.addWidget(self.reset_button)

        self.save_button = QPushButton("Save Processed Image")
        self.save_button.clicked.connect(self.save_processed_image)
        actions_layout.addWidget(self.save_button)

        panel_layout.addLayout(actions_layout)

        # --- Status line ---
        self.status_label = QLabel("Please load an image.")
        self.status_label.setWordWrap(True)
        panel_layout.addWidget(self.status_label)

        # Wrap the panel in a scroll area with a fixed width
        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(PANEL_WIDTH)
        scroll.setFrameShape(QFrame.StyledPanel)
        scroll.setHorizontalScrollBarPolicy(1)  # Qt.ScrollBarAlwaysOff

        root.addWidget(scroll)

        # ---------------- Right: image canvases (2x2) ----------------
        images_layout = QGridLayout()
        images_layout.setSpacing(6)

        self.original_canvas = ImageCanvas("Original Image")
        self.processed_canvas = ImageCanvas("Processed Image")
        self.phase_canvas = ImageCanvas("FFT - Phase")
        self.magnitude_canvas = ImageCanvas("Filtered Magnitude")

        images_layout.addWidget(self.original_canvas, 0, 0)
        images_layout.addWidget(self.processed_canvas, 0, 1)
        images_layout.addWidget(self.magnitude_canvas, 1, 0)
        images_layout.addWidget(self.phase_canvas, 1, 1)

        images_layout.setRowStretch(0, 1)
        images_layout.setRowStretch(1, 1)
        images_layout.setColumnStretch(0, 1)
        images_layout.setColumnStretch(1, 1)

        root.addLayout(images_layout, 1)  # stretch=1 -> takes all remaining width

    def _update_freq_controls(self, kind):
        self.order.setEnabled(kind == "butterworth")

    def _clear_outputs(self):
        """Clear every result canvas so nothing from a previous operation stays on screen."""
        self.processed_canvas.clear()
        self.magnitude_canvas.clear()
        self.phase_canvas.clear()

    # ------------------------------------------------------------------
    # Logic
    # ------------------------------------------------------------------
    def load_image_file(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )

        if not file_path:
            return

        try:
            self.original_image = load_image(file_path)
            self.processed_image = None

            self.original_canvas.show_image(
                self.original_image,
                "Original Image"
            )

            self.processed_canvas.clear()
            self.phase_canvas.clear()
            self.magnitude_canvas.clear()

            self.status_label.setText("Image loaded successfully.")

        except Exception as e:
            show_error(self, e, "Image Loading Error")

    def apply_spatial_filter(self):

        if self.original_image is None:
            show_error(self, "Please load an image first.")
            return

        try:
            filter_name = self.filter_combo.currentData()
            size = int(self.kernel_size.currentText())
            sigma = self.sigma.value()

            self.processed_image = spatial_filter(
                self.original_image,
                name=filter_name,
                size=size,
                sigma=sigma
            )

            self._clear_outputs()

            self.processed_canvas.show_image(
                self.processed_image,
                f"Processed Image - {filter_name}"
            )

            self.status_label.setText(
                f"{filter_name.capitalize()} filter applied successfully."
            )

        except Exception as e:
            show_error(self, e, "Filter Error")

    def analyze_fft(self):

        if self.original_image is None:
            show_error(self, "Please load an image first.")
            return

        try:
            F_shifted, magnitude, phase = image_fft(
                self.original_image
            )

            self._clear_outputs()

            self.processed_canvas.show_image(
                magnitude,
                "FFT - Log Magnitude"
            )

            self.phase_canvas.show_image(
                phase,
                "FFT - Phase",
                cmap="twilight"
            )

            self.status_label.setText(
                "FFT analysis completed successfully."
            )

        except Exception as e:
            show_error(self, e, "FFT Error")

    def apply_frequency_filter(self):

        if self.original_image is None:
            show_error(self, "Please load an image first.")
            return

        try:
            filter_type = self.freq_type.currentData()
            kind = self.freq_kind.currentText()
            cutoff = self.cutoff.value()
            order = self.order.value()

            output, magnitude, phase = frequency_filter(
                self.original_image,
                filter_type=filter_type,
                cutoff=cutoff,
                kind=kind,
                order=order
            )

            self.processed_image = output

            self.processed_canvas.show_image(
                output,
                f"Frequency Filter - {filter_type}"
            )

            self.magnitude_canvas.show_image(
                magnitude,
                "Filtered Log Magnitude"
            )

            self.phase_canvas.show_image(
                phase,
                "Filtered Phase",
                cmap="twilight"
            )

            self.status_label.setText(
                f"{kind.capitalize()} {filter_type} applied successfully."
            )

        except Exception as e:
            show_error(self, e, "Frequency Filter Error")

    def add_noise(self):

        if self.original_image is None:
            show_error(self, "Please load an image first.")
            return

        try:
            self.noisy_image = add_periodic_noise(
                self.original_image,
                amplitude=self.noise_amplitude.value(),
                fx=self.noise_fx.value(),
                fy=self.noise_fy.value()
            )

            self.processed_image = self.noisy_image

            self._clear_outputs()

            self.processed_canvas.show_image(
                self.noisy_image,
                "Image with Periodic Noise"
            )

            _, log_magnitude = analyze_noisy_image(self.noisy_image)

            self.magnitude_canvas.show_image(
                log_magnitude,
                "Noisy Image - FFT Spectrum"
            )

            self.status_label.setText(
                "Periodic noise added successfully."
            )

        except Exception as e:
            show_error(self, e, "Noise Error")

    def remove_noise(self):

        if self.noisy_image is None:
            show_error(self, "Please add periodic noise first.")
            return

        try:
            restored, mask = remove_periodic_noise(
                self.noisy_image,
                fx=self.noise_fx.value(),
                fy=self.noise_fy.value(),
                radius=self.notch_radius.value()
            )

            self.processed_image = restored

            self._clear_outputs()

            self.processed_canvas.show_image(
                restored,
                "Restored Image"
            )

            _, restored_magnitude, _ = image_fft(restored)

            self.magnitude_canvas.show_image(
                restored_magnitude,
                "Restored Image - FFT Spectrum"
            )

            mae, rmse, max_error, psnr = calculate_metrics(
                restored,
                self.original_image
            )

            self.status_label.setText(
                f"Restored | RMSE: {rmse:.2f} | PSNR: {psnr:.2f} dB"
            )

        except Exception as e:
            show_error(self, e, "Noise Removal Error")

    def reset_image(self):

        if self.original_image is None:
            show_error(self, "Please load an image first.")
            return

        self.processed_image = None

        self.original_canvas.show_image(
            self.original_image,
            "Original Image"
        )

        self.processed_canvas.clear()
        self.phase_canvas.clear()
        self.magnitude_canvas.clear()

        self.status_label.setText("Image reset successfully.")

    def save_processed_image(self):

        if self.processed_image is None:
            show_error(self, "No processed image to save.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Processed Image",
            "",
            "PNG Image (*.png);;JPEG Image (*.jpg)"
        )

        if not file_path:
            return

        try:
            from PIL import Image

            image = self.processed_image.clip(0, 255).astype("uint8")
            Image.fromarray(image).save(file_path)

            self.status_label.setText("Image saved successfully.")

        except Exception as e:
            show_error(self, e, "Save Error")