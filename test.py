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


# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image

# # Load a grayscale image
# image = Image.open("./images/lena.bmp").convert("L")  # Convert to grayscale
# image_array = np.array(image)

# #Apply 2D FFT direct
# dft_result = np.fft.fft2(image_array)

# # Shift zero frequency to the center
# dft_result_shifted = np.fft.fftshift(dft_result)

# # Compute magnitude spectrum
# magnitude_spectrum = np.log(1 + np.abs(dft_result_shifted))

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

# import numpy as np
# import cv2
# from matplotlib import pyplot as plt

# def create_low_pass_filter(shape, cutoff):
#     """Create a low-pass filter mask."""
#     N, M = shape
#     center = (N // 2, M // 2)
#     mask = np.zeros((N, M))
#     for x in range(N):
#         for y in range(M):
#             if np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2) <= cutoff:
#                 mask[x, y] = 1
#     return mask

# # Wczytaj obraz lena
# image = cv2.imread('images/lena.bmp', cv2.IMREAD_GRAYSCALE)

# # Uzyskaj kształt obrazu
# shape = image.shape

# # Ustaw częstotliwość odcięcia
# cutoff = 10

# # Utwórz maskę filtra dolnoprzepustowego
# low_pass_filter = create_low_pass_filter(shape, cutoff)

# # Przekształć obraz do przestrzeni częstotliwościowej
# dft = np.fft.fft2(image)
# dft_shifted = np.fft.fftshift(dft)

# # Zastosuj maskę filtra
# filtered_dft = dft_shifted * low_pass_filter

# # Przekształć z powrotem do przestrzeni czasowej
# filtered_dft_shifted = np.fft.ifftshift(filtered_dft)
# filtered_image = np.fft.ifft2(filtered_dft_shifted)
# filtered_image = np.abs(filtered_image)

# # Wyświetl oryginalny i przefiltrowany obraz
# plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Oryginalny obraz')
# plt.subplot(122), plt.imshow(filtered_image, cmap='gray'), plt.title('Przefiltrowany obraz')
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time 

def fft_recursive(signal):
    N = len(signal)
    if N <= 1:  
        return signal
    even = fft_recursive(signal[0::2])
    odd = fft_recursive(signal[1::2])
    combined = [0] * N
    for k in range(N // 2):
        t = np.exp(-2j * np.pi * k / N) * odd[k]
        combined[k] = even[k] + t
        # print("Even", combined[k])
        combined[k + N // 2] = even[k] - t
        # print("Odd", combined[k + N // 2])
    return combined

def ifft_recursive(signal):
    N = len(signal)
    if N <= 1: 
        return signal
    even = ifft_recursive(signal[0::2])
    odd = ifft_recursive(signal[1::2])
    combined = [0] * N
    for k in range(N // 2):
        t = np.exp(2j * np.pi * k / N) * odd[k]
        combined[k] = (even[k] + t) / 2
        combined[k + N // 2] = (even[k] - t) / 2
    return combined


def fft_2d(img):
    rows_fft = np.array([fft_recursive(row) for row in img])
    return np.array([fft_recursive(col) for col in rows_fft.T]).T

def ifft_2d(F):
    rows_ifft = np.array([ifft_recursive(row) for row in F])
    return np.array([ifft_recursive(col) for col in rows_ifft.T]).T

def fftshift_2d(img):
    rows, cols = img.shape
    
    row_mid, col_mid = rows // 2, cols // 2

    temp = np.zeros_like(img)
    # top-left <-> bottom-right
    temp[:row_mid, :col_mid] = img[row_mid:, col_mid:]
    temp[row_mid:, col_mid:] = img[:row_mid, :col_mid]
    # top-right <-> bottom-left
    temp[:row_mid, col_mid:] = img[row_mid:, :col_mid]
    temp[row_mid:, :col_mid] = img[:row_mid, col_mid:]

    return temp
  
     
    
test_matrix = np.array([[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16]])

# Wywołaj funkcję fftshift_2d na przykładowej macierzy
# shifted_matrix = fftshift_2d(test_matrix)
# shifted_matrix1 = np.fft.fftshift(test_matrix)
# Wyświetl oryginalną i przesuniętą macierz

image = Image.open("./images/lena.bmp").convert("L")

# Zmień rozmiar obrazu na 512x512 pikseli
# new_size = (512, 512)
# resized_image = image.resize(new_size)
image_array = np.array(image)

dft_result = fft_2d(image_array)
# shifted_matrix1 = np.fft.fftshift(dft_result)
shifted_matrix = fftshift_2d(dft_result)

dft_magnitude = np.abs(dft_result)
shifted_magnitude = np.abs(shifted_matrix)

print("Oryginalna macierz:")
# print(test_matrix)

print("\nPrzesunięta macierz moim:")
print(shifted_matrix)

print("\nPrzesunięta macierz numpy:")
# print(shifted_matrix1)
plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(np.log1p(dft_magnitude), cmap='gray'), plt.title('Original FFT Magnitude Spectrum')
plt.subplot(122), plt.imshow(np.log1p(shifted_magnitude), cmap='gray'), plt.title('Shifted FFT Magnitude Spectrum')
plt.colorbar()
plt.show()

# def fftshift_manual(spectrum):
#     rows, cols = len(spectrum), len(spectrum[0])  # Dimensions of the spectrum
#     half_rows, half_cols = rows // 2, cols // 2

#     # Create an empty array of the same shape to store the shifted spectrum
#     shifted = [[0j for _ in range(cols)] for _ in range(rows)]

#     # Rearrange the quadrants
#     for i in range(rows):
#         for j in range(cols):
#             # Compute new indices (modulo to handle odd dimensions)
#             new_i = (i + half_rows) % rows
#             new_j = (j + half_cols) % cols
#             shifted[new_i][new_j] = spectrum[i][j]
    
#     return shifted


# # Wczytaj obraz lena i przekonwertuj na skalę szarości
# image = Image.open("./images/lena.bmp").convert("L")

# # Zmień rozmiar obrazu na 512x512 pikseli
# new_size = (512, 512)
# resized_image = image.resize(new_size)
# image_array = np.array(resized_image)

# # Zmierz czas wykonania FFT
# start_time = time.time()

# # Oblicz DFT obrazu za pomocą funkcji fft_2d
# dft_result = fft_2d(image_array)

# # Zmierz czas wykonania FFT
# end_time = time.time()
# execution_time = end_time - start_time
# print(f"Execution time for FFT: {execution_time:.4f} seconds")

# # Zmierz czas wykonania IFFT
# start_time = time.time()

# # Oblicz odwrotną DFT za pomocą funkcji ifft_2d
# reconstructed_image = ifft_2d(dft_result)

# # Zmierz czas wykonania IFFT
# end_time = time.time()
# execution_time = end_time - start_time
# print(f"Execution time for IFFT: {execution_time:.4f} seconds")

# # Oblicz moduł odwrotnej DFT do wizualizacji
# reconstructed_image_real = np.abs(reconstructed_image)

# # Wyświetl oryginalny obraz i obraz po przekształceniach FFT i IFFT
# plt.figure(figsize=(10, 5))
# plt.subplot(121), plt.imshow(image_array, cmap='gray'), plt.title('Original Image')
# plt.subplot(122), plt.imshow(reconstructed_image_real, cmap='gray'), plt.title('After fft and ifft Image')
# plt.show()

