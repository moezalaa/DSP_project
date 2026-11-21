"""Image signal module (assumed exports: the team's __init__.py was not sent, scripts import these four)."""
from .processing import load_image, spatial_filter, image_fft, frequency_filter

__all__ = ["load_image", "spatial_filter", "image_fft", "frequency_filter"]
