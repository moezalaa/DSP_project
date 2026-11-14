
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFileDialog,
    QComboBox, QSpinBox, QDoubleSpinBox
)

from core.gui_widgets import ImageCanvas, show_error
from .processing import load_image, spatial_filter, image_fft, frequency_filter


class ImageSignalTab(QWidget):

    def __init__(self):
        super().__init__()

        self.original_image = None
        self.processed_image = None

        self.init_ui()

    def init_ui(self):

        main_layout = QVBoxLayout(self)

        # Title
        title = QLabel("Image Signal Processing")
        main_layout.addWidget(title)

        # Load Image
        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image_file)
        main_layout.addWidget(self.load_button)

        # fft_button
        self.fft_button = QPushButton("Analyze FFT")
        self.fft_button.clicked.connect(self.analyze_fft)

        main_layout.addWidget(self.fft_button)

        # reset_button
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_image)
        main_layout.addWidget(self.reset_button)

        # save_button
        self.save_button = QPushButton("Save Processed Image")
        self.save_button.clicked.connect(self.save_processed_image)
        main_layout.addWidget(self.save_button)

        # Filter controls
        filter_layout = QHBoxLayout()

        filter_layout.addWidget(QLabel("Spatial Filter:"))

        self.filter_combo = QComboBox()
        self.filter_combo.addItem("Box", "box")
        self.filter_combo.addItem("Gaussian", "gaussian")
        self.filter_combo.addItem("Sharpen", "sharpen")
        self.filter_combo.addItem("Sobel X", "sobel_x")
        self.filter_combo.addItem("Sobel Y", "sobel_y")
        self.filter_combo.addItem("Sobel", "sobel")
        self.filter_combo.addItem("Prewitt", "prewitt")
        self.filter_combo.addItem("Laplacian", "laplacian")

        filter_layout.addWidget(self.filter_combo)

        filter_layout.addWidget(QLabel("Kernel Size:"))

        self.kernel_size = QComboBox()
        self.kernel_size.addItems(["3", "5", "7", "9"])

        filter_layout.addWidget(self.kernel_size)

        filter_layout.addWidget(QLabel("Sigma:"))

        self.sigma = QDoubleSpinBox()
        self.sigma.setRange(0.1, 10.0)
        self.sigma.setSingleStep(0.1)
        self.sigma.setValue(1.0)

        filter_layout.addWidget(self.sigma)

        self.apply_button = QPushButton("Apply Filter")
        self.apply_button.clicked.connect(self.apply_spatial_filter)

        filter_layout.addWidget(self.apply_button)

        main_layout.addLayout(filter_layout)

        # Frequency Filter Controls
        freq_layout = QHBoxLayout()

        freq_layout.addWidget(QLabel("Frequency Filter:"))

        self.freq_type = QComboBox()
        self.freq_type.addItem("Low Pass", "lowpass")
        self.freq_type.addItem("High Pass", "highpass")
        freq_layout.addWidget(self.freq_type)

        freq_layout.addWidget(QLabel("Filter Type:"))

        self.freq_kind = QComboBox()
        self.freq_kind.addItems(["ideal", "gaussian", "butterworth"])
        freq_layout.addWidget(self.freq_kind)

        freq_layout.addWidget(QLabel("Cutoff:"))

        self.cutoff = QDoubleSpinBox()
        self.cutoff.setRange(0.1, 500)
        self.cutoff.setValue(30)
        self.cutoff.setSingleStep(1)
        freq_layout.addWidget(self.cutoff)

        freq_layout.addWidget(QLabel("Order:"))

        self.order = QSpinBox()
        self.order.setRange(1, 10)
        self.order.setValue(2)
        freq_layout.addWidget(self.order)

        self.freq_button = QPushButton("Apply Frequency Filter")
        self.freq_button.clicked.connect(self.apply_frequency_filter)
        freq_layout.addWidget(self.freq_button)

        main_layout.addLayout(freq_layout)

        # Image Canvases
        images_layout = QGridLayout()

        self.original_canvas = ImageCanvas("Original Image")
        self.processed_canvas = ImageCanvas("Processed Image")
        self.phase_canvas = ImageCanvas("FFT - Phase")
        self.magnitude_canvas = ImageCanvas("Filtered Magnitude")

        images_layout.addWidget(self.original_canvas, 0, 0)
        images_layout.addWidget(self.processed_canvas, 0, 1)
        images_layout.addWidget(self.magnitude_canvas, 1, 0)
        images_layout.addWidget(self.phase_canvas, 1, 1)

        main_layout.addLayout(images_layout)

        # Status
        self.status_label = QLabel("Please load an image.")
        main_layout.addWidget(self.status_label)

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