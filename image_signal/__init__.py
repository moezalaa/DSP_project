"""image_signal module (Day 3+)."""

from .processing import (
    load_image,
    spatial_filter,
    image_fft,
    frequency_filter
)

__all__ = [
    "load_image",
    "spatial_filter",
    "image_fft",
    "frequency_filter"
]
