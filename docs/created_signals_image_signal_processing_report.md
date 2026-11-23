# Report: Created Signals -- Image Signal Processing

## 1. Introduction

Image Signal Processing is concerned with analyzing and modifying
digital images to improve their quality or extract useful information.
One common problem is **periodic noise**, which appears as repeated
patterns or stripes across an image.

This project demonstrates how periodic noise can be generated, analyzed
in the frequency domain, and reduced using a notch filter. The
implementation also provides a graphical interface for applying the
process and evaluating the results.

## 2. Created Signal: Periodic Noise

The project creates a periodic noise signal and adds it to a grayscale
image. The noise is generated using a two-dimensional sinusoidal
function:

\[
N(x,y)=A`\sin`{=tex}`\left`{=tex}(2`\pi`{=tex}`\left`{=tex}(`\frac{f_x x}{W}`{=tex}+`\frac{f_y y}{H}`{=tex}`\right`{=tex})`\right`{=tex})
\]

Where:

-   \(A\) is the noise amplitude.
-   (f_x) and (f_y) are the horizontal and vertical frequencies.
-   \(W\) and (H) are the image width and height.
-   \(x\) and (y) represent pixel coordinates.

The noisy image is produced by adding the noise signal to the original
grayscale image:

\[
I\_{`\text{noisy}`{=tex}}(x,y)=I\_{`\text{original}`{=tex}}(x,y)+N(x,y)
\]

The default parameters used by the implementation are an amplitude of
30, (f_x=8), and (f_y=6). The generated result is kept as a
floating-point array without clipping, so the original image is not
modified.

## 3. Frequency-Domain Analysis Using FFT

After adding periodic noise, the image is analyzed using the **2D Fast
Fourier Transform (FFT)**. The FFT converts the image from the spatial
domain into the frequency domain.

The implementation shifts the zero-frequency component to the center of
the spectrum and displays the logarithm of the magnitude. This makes
frequency components easier to inspect.

Periodic noise produces distinct peaks in the frequency spectrum. These
peaks appear as a symmetric pair around the center because the input
image is real-valued. Identifying their locations helps determine which
frequencies should be removed.

## 4. Noise Removal Using a Notch Filter

The project uses a **notch filter mask** to suppress the frequency
components associated with periodic noise.

The mask is initialized with ones, meaning that most frequency
components are preserved. Two circular regions are then set to zero
around the detected noise peaks, one on each side of the spectrum. The
radius controls the size of the removed regions.

The restoration process follows these steps:

1.  Calculate the FFT of the noisy image.
2.  Shift the zero-frequency component to the center.
3.  Multiply the shifted spectrum by the notch mask.
4.  Apply the inverse shift.
5.  Apply the inverse FFT and take the real component to obtain the
    restored image.

The mask is symmetric and leaves the central DC component unchanged.
Choosing an appropriate radius is important: a radius that is too large
may remove useful image information in addition to the noise.

## 5. Image Quality Evaluation

The project evaluates the image before and after noise removal using
several error and quality metrics:

-   **MAE (Mean Absolute Error):** measures the average absolute
    difference between two images.
-   **MSE (Mean Squared Error):** measures the average squared
    difference between corresponding pixels.
-   **RMSE (Root Mean Squared Error):** the square root of MSE,
    expressed on the same scale as pixel values.
-   **Maximum Error:** records the largest absolute pixel difference.
-   **PSNR (Peak Signal-to-Noise Ratio):** expresses image quality in
    decibels; a higher PSNR generally indicates less error relative to
    the reference image.

The tests compare the noisy and restored images with the original image
to check whether the filtering process improves image quality. They also
check that using incorrect frequency locations or an excessively large
notch radius can lead to poorer results.

## 6. Graphical User Interface

The graphical interface allows the user to control the main parameters
of the process, including:

-   Noise amplitude.
-   Horizontal frequency (f_x).
-   Vertical frequency (f_y).
-   Notch filter radius.

The interface displays the noisy image and its frequency spectrum, then
shows the restored image, the corresponding spectra, and the notch mask.
It also presents image-quality measurements before and after filtering,
making the effect of the process easier to compare.

## 7. Testing

The project includes tests for the main processing functions and the
graphical interface. The tests check that:

-   The generated noise follows the specified sinusoidal equation.
-   A zero noise amplitude leaves the image unchanged.
-   The input image is not modified during noise generation.
-   The frequency spectrum contains the expected symmetric peaks.
-   The notch mask has the required symmetry and radius behavior.
-   The restoration output matches the masked inverse-FFT process.
-   The quality metrics indicate improvement when the correct noise
    frequencies are removed.
-   The interface displays the processing results and quality
    measurements.

These tests help verify both the mathematical operations and the way the
user interacts with the application.

## 8. Conclusion

The project demonstrates a complete workflow for handling periodic noise
in digital images. It creates a controlled sinusoidal noise signal,
identifies its frequency components using FFT, and reduces the noise
with a symmetric notch filter. Image-quality metrics and the graphical
interface help assess and visualize the results.

Overall, the work shows how frequency-domain processing can be used to
detect and suppress periodic noise while preserving as much useful image
information as possible.
