# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image

# # Define the dft2d_slow function (your provided code).

# def dft2d_slow(image):
#     rows, cols = image.shape
#     result = np.zeros((rows, cols), dtype=complex)
#     for u in range(rows):
#         for v in range(cols):
#             sum_pixel = 0
#             for x in range(rows):
#                 for y in range(cols):
#                     sum_pixel += image[x, y] * np.exp(-2j * np.pi * ((u * x / rows) + (v * y / cols)))
#             result[u, v] = sum_pixel
#     return result

# # Load a grayscale image
# image = Image.open("./images/lena.bmp").convert("L")  # Convert to grayscale
# image_array = np.array(image)

# # Apply 2D DFT
# dft_result = dft2d_slow(image_array)

# # Compute magnitude spectrum
# magnitude_spectrum = np.log(1 + np.abs(dft_result))

# # Plot the original image and magnitude spectrum
# plt.figure(figsize=(12, 6))

# # Original Image
# plt.subplot(1, 2, 1)
# plt.title("Original Image")
# plt.imshow(image_array, cmap="gray")
# plt.axis("off")

# # Magnitude Spectrum
# plt.subplot(1, 2, 2)
# plt.title("Magnitude Spectrum")
# plt.imshow(magnitude_spectrum, cmap="gray")
# plt.axis("off")

# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load a grayscale image
image = Image.open("./images/lena.bmp").convert("L")  # Convert to grayscale
image_array = np.array(image)

#Apply 2D FFT direct
dft_result = np.fft.fft2(image_array)

# Shift zero frequency to the center
dft_result_shifted = np.fft.fftshift(dft_result)

# Compute magnitude spectrum
magnitude_spectrum = np.log(1 + np.abs(dft_result_shifted))

# Plot the original image and magnitude spectrum
plt.figure(figsize=(12, 6))

# Original Image
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image_array, cmap="gray")
plt.axis("off")

# Magnitude Spectrum
plt.subplot(1, 2, 2)
plt.title("Magnitude Spectrum")
plt.imshow(magnitude_spectrum, cmap="gray")
plt.axis("off")

plt.show()

#Apply 2D FFT inverse

# idft_result = np.fft.ifft2(image_array)

# # Shift zero frequency to the center
# idft_result_shifted = np.fft.ifftshift(idft_result)

# # Compute magnitude spectrum
# magnitude_spectrum = np.log(1 + np.abs(idft_result_shifted))

# # Plot the original image and magnitude spectrum
# plt.figure(figsize=(12, 6))

# # Original Image
# plt.subplot(1, 2, 1)
# plt.title("Original Image")
# plt.imshow(image_array, cmap="gray")
# plt.axis("off")

# # Magnitude Spectrum
# plt.subplot(1, 2, 2)
# plt.title("Magnitude Spectrum")
# plt.imshow(magnitude_spectrum, cmap="gray")
# plt.axis("off")

# plt.show()
