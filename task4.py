# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# import time

# def dft_2d(image):
#     """Compute the 2D Discrete Fourier Transform (slow version)."""
#     N, M = image.shape
#     dft = np.zeros((N, M), dtype=complex)
#     for u in range(N):
#         for v in range(M):
#             sum_val = 0
#             for x in range(N):
#                 for y in range(M):
#                     sum_val += image[x, y] * np.exp(-2j * np.pi * (u * x / N + v * y / M))
#             dft[u, v] = sum_val
#     return dft

# def idft_2d(dft):
#     """Compute the 2D Inverse Discrete Fourier Transform (slow version)."""
#     N, M = dft.shape
#     idft = np.zeros((N, M), dtype=complex)
#     for x in range(N):
#         for y in range(M):
#             sum_val = 0
#             for u in range(N):
#                 for v in range(M):
#                     sum_val += dft[u, v] * np.exp(2j * np.pi * (u * x / N + v * y / M))
#             idft[x, y] = sum_val / (N * M)
#     return idft

# def fft_recursive(signal):
#     N = len(signal)
#     if N <= 1:  
#         return signal
#     even = fft_recursive(signal[0::2])
#     odd = fft_recursive(signal[1::2])
#     combined = [0] * N
#     for k in range(N // 2):
#         t = np.exp(-2j * np.pi * k / N) * odd[k]
#         combined[k] = even[k] + t
#         combined[k + N // 2] = even[k] - t
#     return combined

# def ifft_recursive(signal):
#     N = len(signal)
#     if N <= 1: 
#         return signal
#     even = ifft_recursive(signal[0::2])
#     odd = ifft_recursive(signal[1::2])
#     combined = [0] * N
#     for k in range(N // 2):
#         t = np.exp(2j * np.pi * k / N) * odd[k]
#         combined[k] = (even[k] + t) / 2
#         combined[k + N // 2] = (even[k] - t) / 2
#     return combined

# def fftshift_2d(img):
#     rows, cols = img.shape
    
#     row_mid, col_mid = rows // 2, cols // 2

#     temp = np.zeros_like(img)
#     # top-left <-> bottom-right
#     temp[:row_mid, :col_mid] = img[row_mid:, col_mid:]
#     temp[row_mid:, col_mid:] = img[:row_mid, :col_mid]
#     # top-right <-> bottom-left
#     temp[:row_mid, col_mid:] = img[row_mid:, :col_mid]
#     temp[row_mid:, :col_mid] = img[:row_mid, col_mid:]

#     return temp

# def ifftshift_2d(img):
#     rows, cols = img.shape
    
#     row_mid, col_mid = rows // 2, cols // 2

#     temp = np.zeros_like(img)
#     # bottom-right <-> top-left
#     temp[row_mid:, col_mid:] = img[:row_mid, :col_mid]
#     temp[:row_mid, :col_mid] = img[row_mid:, col_mid:]
#     # bottom-left <-> top-right
#     temp[row_mid:, :col_mid] = img[:row_mid, col_mid:]
#     temp[:row_mid, col_mid:] = img[row_mid:, :col_mid]

#     return temp

# def fft_2d(img):
#     rows_fft = np.array([fft_recursive(row) for row in img])
#     return np.array([fft_recursive(col) for col in rows_fft.T]).T

# def ifft_2d(F):
#     rows_ifft = np.array([ifft_recursive(row) for row in F])
#     return np.array([ifft_recursive(col) for col in rows_ifft.T]).T

# def visualize_spectrum(image, dft, title="Spectrum", save=False):
#     """Visualize the original image and the magnitude spectrum of a Fourier transform."""
#     magnitude = np.log(np.abs(dft) + 1)  # Apply log for better visualization
#     plt.figure(figsize=(12, 6))
    
#     # Original image
#     plt.subplot(1, 2, 1)
#     plt.title("Original Image")
#     plt.imshow(image, cmap='gray')
#     plt.axis('off')  # Hide axes
    
#     # Magnitude spectrum
#     plt.subplot(1, 2, 2)
#     plt.title(f"{title} - Magnitude Spectrum")
#     plt.imshow(magnitude, cmap='gray')
#     plt.axis('off')  # Hide axes
    
#     if save:
#         plt.savefig(f"{title}_spectrum.png")
#     plt.show()  # Show the plot

# def apply_filter(dft, filter_mask):
#     """Apply a filter mask to the Fourier transform."""
#     return dft * filter_mask

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

# def create_high_pass_filter(shape, cutoff):
#     """Create a high-pass filter mask."""
#     return 1 - create_low_pass_filter(shape, cutoff)

# def create_band_pass_filter(shape, low_cutoff, high_cutoff):
#     """Create a band-pass filter mask."""
#     low_pass = create_low_pass_filter(shape, high_cutoff)
#     high_pass = create_high_pass_filter(shape, low_cutoff)
#     return low_pass * high_pass

# def create_band_cut_filter(shape, low_cutoff, high_cutoff):
#     """Create a band-cut filter mask."""
#     return 1 - create_band_pass_filter(shape, low_cutoff, high_cutoff)

# def high_pass_directional_filter_mask(shape, direction='horizontal', width=10):
#     """Create a high-pass directional filter mask."""
#     rows, cols = shape
#     crow, ccol = rows // 2, cols // 2  # Center of the mask

#     mask = np.zeros((rows, cols), dtype=np.float32)
#     if direction == 'horizontal':
#         mask[crow - width:crow + width, :] = 1  # Horizontal passband
#     elif direction == 'vertical':
#         mask[:, ccol - width:ccol + width] = 1  # Vertical passband

#     return mask

# def high_pass_directional_filter_mask(shape, mask_path):
#     """Create a high-pass directional filter mask using an image mask."""
#     mask_image = Image.open(mask_path).convert("L")
#     mask = np.array(mask_image).astype(np.float32) / 255.0  # Normalize to [0, 1]
#     mask_resized = np.array(Image.fromarray(mask).resize(shape[::-1], Image.BILINEAR))
#     return mask_resized

# def phase_modifying_filter_mask(shape, k=1, l=1):
#     """Create a phase-modifying filter mask."""
#     rows, cols = shape
#     mask = np.zeros((rows, cols), dtype=np.complex64)

#     for n in range(rows):
#         for m in range(cols):
#             exponent = -1j * ((k * n * 2 * np.pi / rows) + (l * m * 2 * np.pi / cols) + ((k + l) * np.pi))
#             mask[n, m] = np.exp(exponent)

#     return mask

# def process_channel(image_channel, rows, cols, channel=None):
#     """Process a single channel of an image using 2D FFT and filtering."""
#     if channel is not None:
#         print(f"Processing channel {channel}...")
    
#     dft = fft_2d(image_channel)
#     dft_shifted = fftshift_2d(dft)
#     if channel is not None:
#         visualize_spectrum(image_channel, dft_shifted, title=f"Original Spectrum Channel {channel}")
#     else:
#         visualize_spectrum(image_channel, dft_shifted, title="Original Spectrum")
    
#     # Example: Low-pass filter
#     # Choose one filter to apply (comment/uncomment to switch)
#     filter_mask = create_low_pass_filter((rows, cols), cutoff=90)  # Example: Low-pass filter
#     # filter_mask = create_high_pass_filter((rows, cols), cutoff=50)  # High-pass filter
#     # filter_mask = create_band_pass_filter((rows, cols), low_cutoff=5, high_cutoff=300)  # Band-pass filter
#     # filter_mask = create_band_cut_filter((rows, cols), low_cutoff=30, high_cutoff=100)  # Band-cut filter
#     # filter_mask = high_pass_directional_filter_mask((rows, cols), direction='horizontal', width=10)  # Horizontal edges
#     # filter_mask = high_pass_directional_filter_mask((rows, cols), filter_mask_path)
#     # filter_mask = high_pass_directional_filter_mask((rows, cols), direction='vertical', width=10)  # Vertical edges
#     # filter_mask = phase_modifying_filter_mask((rows, cols), k=100, l=100)  # Phase-modifying filter

#     # Example: Use dft_2d function
#     dft_result = fft_2d(image_channel)
#     visualize_spectrum(image_channel, dft_result, title="DFT Spectrum")

#     # Apply the filter
#     filtered_dft = apply_filter(dft_shifted, filter_mask)

#     if channel is not None:
#         visualize_spectrum(image_channel, filtered_dft, title=f"Filtered Spectrum Channel {channel}")
#     else:
#         visualize_spectrum(image_channel, filtered_dft, title="Filtered Spectrum")

#     filtered_dft_unshifted = ifftshift_2d(filtered_dft)
#     filtered_channel = np.real(ifft_2d(filtered_dft_unshifted))
#     return filtered_channel, np.abs(dft_shifted)

# def combine_spectrums(spectrums):
#     """Combine three magnitude spectrums into one."""
#     combined_spectrum = np.sqrt(np.sum(np.array(spectrums) ** 2, axis=0))
#     return combined_spectrum

# def main():
#     # image = np.array(Image.open('./images/lenac.bmp')) 
#     image = np.array(Image.open('filtered_image.bmp')) 

#     is_color = len(image.shape) == 3 

#     if is_color:
#         rows, cols, _ = image.shape
#         filtered_image = np.zeros_like(image, dtype=np.float32)
#         spectrums = []

#         for channel in range(3): 
#             filtered_channel, spectrum = process_channel(
#                 image[:, :, channel], rows, cols, channel=channel
#             )
#             filtered_image[:, :, channel] = filtered_channel
#             spectrums.append(spectrum)

#         combined_spectrum = combine_spectrums(spectrums)
#         visualize_spectrum(image, combined_spectrum, title="Combined Spectrum")
#     else:
#         rows, cols = image.shape
#         filtered_image, spectrum = process_channel(image, rows, cols)
#         visualize_spectrum(image, spectrum, title="Spectrum")

#     filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
#     if is_color:
#         filtered_image_pil = Image.fromarray(filtered_image, mode='RGB')
#     else:
#         filtered_image_pil = Image.fromarray(filtered_image)
#     filtered_image_pil.save("filtered_image.bmp")
#     print("Filtered image saved as 'filtered_image.bmp'.")

# if __name__ == "__main__":
#     main()

# # import numpy as np
# # import matplotlib.pyplot as plt
# # from PIL import Image
# # import time

# # def dft_2d(image):
# #     """Compute the 2D Discrete Fourier Transform (slow version)."""
# #     N, M = image.shape
# #     dft = np.zeros((N, M), dtype=complex)
# #     for u in range(N):
# #         for v in range(M):
# #             sum_val = 0
# #             for x in range(N):
# #                 for y in range(M):
# #                     sum_val += image[x, y] * np.exp(-2j * np.pi * (u * x / N + v * y / M))
# #             dft[u, v] = sum_val
# #     return dft

# # def idft_2d(dft):
# #     """Compute the 2D Inverse Discrete Fourier Transform (slow version)."""
# #     N, M = dft.shape
# #     idft = np.zeros((N, M), dtype=complex)
# #     for x in range(N):
# #         for y in range(M):
# #             sum_val = 0
# #             for u in range(N):
# #                 for v in range(M):
# #                     sum_val += dft[u, v] * np.exp(2j * np.pi * (u * x / N + v * y / M))
# #             idft[x, y] = sum_val / (N * M)
# #     return idft

# # def fft_recursive(signal):
# #     N = len(signal)
# #     if N <= 1:  
# #         return signal
# #     even = fft_recursive(signal[0::2])
# #     odd = fft_recursive(signal[1::2])
# #     combined = [0] * N
# #     for k in range(N // 2):
# #         t = np.exp(-2j * np.pi * k / N) * odd[k]
# #         combined[k] = even[k] + t
# #         combined[k + N // 2] = even[k] - t
# #     return combined

# # def ifft_recursive(signal):
# #     N = len(signal)
# #     if N <= 1: 
# #         return signal
# #     even = ifft_recursive(signal[0::2])
# #     odd = ifft_recursive(signal[1::2])
# #     combined = [0] * N
# #     for k in range(N // 2):
# #         t = np.exp(2j * np.pi * k / N) * odd[k]
# #         combined[k] = (even[k] + t) / 2
# #         combined[k + N // 2] = (even[k] - t) / 2
# #     return combined

# # def fftshift_2d(img):
# #     rows, cols = img.shape
    
# #     row_mid, col_mid = rows // 2, cols // 2

# #     temp = np.zeros_like(img)
# #     # top-left <-> bottom-right
# #     temp[:row_mid, :col_mid] = img[row_mid:, col_mid:]
# #     temp[row_mid:, col_mid:] = img[:row_mid, :col_mid]
# #     # top-right <-> bottom-left
# #     temp[:row_mid, col_mid:] = img[row_mid:, :col_mid]
# #     temp[row_mid:, :col_mid] = img[:row_mid, col_mid:]

# #     return temp

# # def ifftshift_2d(img):
# #     rows, cols = img.shape
    
# #     row_mid, col_mid = rows // 2, cols // 2

# #     temp = np.zeros_like(img)
# #     # bottom-right <-> top-left
# #     temp[row_mid:, col_mid:] = img[:row_mid, :col_mid]
# #     temp[:row_mid, :col_mid] = img[row_mid:, col_mid:]
# #     # bottom-left <-> top-right
# #     temp[row_mid:, :col_mid] = img[:row_mid, col_mid:]
# #     temp[:row_mid, col_mid:] = img[row_mid:, :col_mid]

# #     return temp

# # def fft_2d(img):
# #     rows_fft = np.array([fft_recursive(row) for row in img])
# #     return np.array([fft_recursive(col) for col in rows_fft.T]).T

# # def ifft_2d(F):
# #     rows_ifft = np.array([ifft_recursive(row) for row in F])
# #     return np.array([ifft_recursive(col) for col in rows_ifft.T]).T

# # def visualize_spectrum(image, dft, title="Spectrum", save=False):
# #     """Visualize the original image and the magnitude spectrum of a Fourier transform."""
# #     magnitude = np.log(np.abs(dft) + 1)  # Apply log for better visualization
# #     plt.figure(figsize=(12, 6))
    
# #     # Original image
# #     plt.subplot(1, 2, 1)
# #     plt.title("Original Image")
# #     plt.imshow(image, cmap='gray')
# #     plt.axis('off')  # Hide axes
    
# #     # Magnitude spectrum
# #     plt.subplot(1, 2, 2)
# #     plt.title(f"{title} - Magnitude Spectrum")
# #     plt.imshow(magnitude, cmap='gray')
# #     plt.axis('off')  # Hide axes
    
# #     if save:
# #         plt.savefig(f"{title}_spectrum.png")
# #     plt.show()  # Show the plot

# # def process_channel(image_channel, rows, cols, channel=None):
# #     """Process a single channel of an image using 2D FFT and return the magnitude spectrum."""
# #     if channel is not None:
# #         print(f"Processing channel {channel}...")
    
# #     dft = fft_2d(image_channel)
# #     dft_shifted = fftshift_2d(dft)
# #     magnitude_spectrum = np.abs(dft_shifted)
    
# #     if channel is not None:
# #         visualize_spectrum(image_channel, dft_shifted, title=f"Original Spectrum Channel {channel}")
# #     else:
# #         visualize_spectrum(image_channel, dft_shifted, title="Original Spectrum")
    
# #     return magnitude_spectrum

# # def combine_spectrums(spectrums):
# #     """Combine three magnitude spectrums into one."""
# #     combined_spectrum = np.sqrt(np.sum(np.array(spectrums) ** 2, axis=0))
# #     return combined_spectrum

# # def main():
# #     image = np.array(Image.open('./images/lenac.bmp')) 
# #     is_color = len(image.shape) == 3 

# #     if is_color:
# #         rows, cols, _ = image.shape
# #         spectrums = []

# #         for channel in range(3): 
# #             spectrum = process_channel(image[:, :, channel], rows, cols, channel=channel)
# #             spectrums.append(spectrum)

# #         combined_spectrum = combine_spectrums(spectrums)
# #         visualize_spectrum(image, combined_spectrum, title="Combined Spectrum")
# #     else:
# #         rows, cols = image.shape
# #         spectrum = process_channel(image, rows, cols)
# #         visualize_spectrum(image, spectrum, title="Spectrum")

# # if __name__ == "__main__":
# #     main()


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time

def dft_2d(image):
    """Compute the 2D Discrete Fourier Transform (slow version)."""
    N, M = image.shape
    dft = np.zeros((N, M), dtype=complex)
    for u in range(N):
        for v in range(M):
            sum_val = 0
            for x in range(N):
                for y in range(M):
                    sum_val += image[x, y] * np.exp(-2j * np.pi * (u * x / N + v * y / M))
            dft[u, v] = sum_val
    return dft

def idft_2d(dft):
    """Compute the 2D Inverse Discrete Fourier Transform (slow version)."""
    N, M = dft.shape
    idft = np.zeros((N, M), dtype=complex)
    for x in range(N):
        for y in range(M):
            sum_val = 0
            for u in range(N):
                for v in range(M):
                    sum_val += dft[u, v] * np.exp(2j * np.pi * (u * x / N + v * y / M))
            idft[x, y] = sum_val / (N * M)
    return idft

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
        combined[k + N // 2] = even[k] - t
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

def ifftshift_2d(img):
    rows, cols = img.shape
    
    row_mid, col_mid = rows // 2, cols // 2

    temp = np.zeros_like(img)
    # bottom-right <-> top-left
    temp[row_mid:, col_mid:] = img[:row_mid, :col_mid]
    temp[:row_mid, :col_mid] = img[row_mid:, col_mid:]
    # bottom-left <-> top-right
    temp[row_mid:, :col_mid] = img[:row_mid, col_mid:]
    temp[:row_mid, col_mid:] = img[row_mid:, :col_mid]

    return temp

def fft_2d(img):
    rows_fft = np.array([fft_recursive(row) for row in img])
    return np.array([fft_recursive(col) for col in rows_fft.T]).T

def ifft_2d(F):
    rows_ifft = np.array([ifft_recursive(row) for row in F])
    return np.array([ifft_recursive(col) for col in rows_ifft.T]).T

def visualize_spectrum(image, dft, title="Spectrum", save=False):
    """Visualize the original image and the magnitude spectrum of a Fourier transform."""
    magnitude = np.log(np.abs(dft) + 1)  # Apply log for better visualization
    plt.figure(figsize=(12, 6))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')
    plt.axis('off')  # Hide axes
    
    # Magnitude spectrum
    plt.subplot(1, 2, 2)
    plt.title(f"{title} - Magnitude Spectrum")
    plt.imshow(magnitude, cmap='gray')
    plt.axis('off')  # Hide axes
    
    if save:
        plt.savefig(f"{title}_spectrum.png")
    plt.show()  # Show the plot

def visualize_filtered_spectrum(image, original_dft, filtered_dft, title="Filtered Spectrum", save=False):
    """Visualize the original and filtered magnitude spectrum of a Fourier transform."""
    original_magnitude = np.log(np.abs(original_dft) + 1)  # Apply log for better visualization
    filtered_magnitude = np.log(np.abs(filtered_dft) + 1)  # Apply log for better visualization
    plt.figure(figsize=(12, 6))
    
    # Original magnitude spectrum
    plt.subplot(1, 2, 1)
    plt.title(f"{title} - Original Magnitude Spectrum")
    plt.imshow(original_magnitude, cmap='gray')
    plt.axis('off')  # Hide axes
    
    # Filtered magnitude spectrum
    plt.subplot(1, 2, 2)
    plt.title(f"{title} - Filtered Magnitude Spectrum")
    plt.imshow(filtered_magnitude, cmap='gray')
    plt.axis('off')  # Hide axes
    
    if save:
        plt.savefig(f"{title}_filtered_spectrum.png")
    plt.show()  # Show the plot

def apply_filter(dft, filter_mask):
    """Apply a filter mask to the Fourier transform."""
    return dft * filter_mask

def create_low_pass_filter(shape, cutoff):
    """Create a low-pass filter mask."""
    N, M = shape
    center = (N // 2, M // 2)
    mask = np.zeros((N, M))
    for x in range(N):
        for y in range(M):
            if np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2) <= cutoff:
                mask[x, y] = 1
    return mask

def create_high_pass_filter(shape, cutoff):
    """Create a high-pass filter mask."""
    return 1 - create_low_pass_filter(shape, cutoff)

def create_band_pass_filter(shape, low_cutoff, high_cutoff):
    """Create a band-pass filter mask."""
    low_pass = create_low_pass_filter(shape, high_cutoff)
    high_pass = create_high_pass_filter(shape, low_cutoff)
    return low_pass * high_pass

def create_band_cut_filter(shape, low_cutoff, high_cutoff):
    """Create a band-cut filter mask."""
    return 1 - create_band_pass_filter(shape, low_cutoff, high_cutoff)



def high_pass_directional_filter_mask(shape, start_angle=30, end_angle=50, cutoff=30):

    rows, cols = shape
    crow, ccol = rows // 2, cols // 2

    mask = np.zeros((rows, cols), dtype=np.float32)
    start_angle_rad = np.radians(start_angle)
    end_angle_rad = np.radians(end_angle)

    if start_angle_rad < 0:
        start_angle_rad += 2 * np.pi
    if end_angle_rad < 0:
        end_angle_rad += 2 * np.pi

    start_angle_rad_opposite = start_angle_rad + np.pi
    end_angle_rad_opposite = end_angle_rad + np.pi

    if start_angle_rad_opposite >= 2 * np.pi:
        start_angle_rad_opposite -= 2 * np.pi
    if end_angle_rad_opposite >= 2 * np.pi:
        end_angle_rad_opposite -= 2 * np.pi

    for x in range(rows):
        for y in range(cols):
            dx = x - crow
            dy = y - ccol
            distance = np.sqrt(dx**2 + dy**2)
            theta = np.arctan2(dy, dx)

            if theta < 0:
                theta += 2 * np.pi

            if distance > cutoff:
                if start_angle_rad <= end_angle_rad:
                    if start_angle_rad <= theta <= end_angle_rad:
                        mask[x, y] = 1.0
                    elif start_angle_rad_opposite <= theta <= end_angle_rad_opposite:
                        mask[x, y] = 1.0
                else:
                    if (start_angle_rad <= theta <= 2 * np.pi) or (0 <= theta <= end_angle_rad):
                        mask[x, y] = 1.0
                    if (start_angle_rad_opposite <= theta <= 2 * np.pi) or (0 <= theta <= end_angle_rad_opposite):
                        mask[x, y] = 1.0

    return mask






def phase_modifying_filter_mask(shape, k=1, l=1):
    """Create a phase-modifying filter mask."""
    rows, cols = shape
    mask = np.zeros((rows, cols), dtype=np.complex64)

    for n in range(rows):
        for m in range(cols):
            exponent = -1j * ((k * n * 2 * np.pi / rows) + (l * m * 2 * np.pi / cols) + ((k + l) * np.pi))
            mask[n, m] = np.exp(exponent)

    return mask

def process_channel(image_channel, rows, cols, filter_mask=None, channel=None):
    """Process a single channel of an image using 2D FFT, apply a filter, and return the magnitude spectrum."""
    if channel is not None:
        print(f"Processing channel {channel}...")
    
    dft = fft_2d(image_channel)
    dft_shifted = fftshift_2d(dft)
    magnitude_spectrum = np.abs(dft_shifted)
    
    if filter_mask is not None:
        filtered_dft = apply_filter(dft_shifted, filter_mask)
        filtered_magnitude_spectrum = np.abs(filtered_dft)
        return magnitude_spectrum, filtered_magnitude_spectrum, filtered_dft
    else:
        return magnitude_spectrum

def combine_spectrums(spectrums):
    """Combine three magnitude spectrums into one."""
    combined_spectrum = np.sqrt(np.sum(np.array(spectrums) ** 2, axis=0))
    return combined_spectrum

def main():
    image = np.array(Image.open('./images/task4/F5test3.png').convert('RGB'))
    # image = np.array(Image.open('./images/lenac.bmp'))
    # image = np.array(Image.open('filtered_image.bmp')) 
    is_color = len(image.shape) == 3 

    if is_color:
        rows, cols, _ = image.shape
        spectrums = []
        filtered_spectrums = []
        filtered_channels = []
        filter_mask = high_pass_directional_filter_mask((rows, cols), start_angle=105, end_angle=125, cutoff=20)

        for channel in range(3): 
            spectrum, filtered_spectrum, filtered_dft = process_channel(image[:, :, channel], rows, cols, filter_mask, channel=channel)
            spectrums.append(spectrum)
            filtered_spectrums.append(filtered_spectrum)
            filtered_channel = np.real(ifft_2d(ifftshift_2d(filtered_dft)))
            filtered_channels.append(filtered_channel)

        combined_spectrum = combine_spectrums(spectrums)
        combined_filtered_spectrum = combine_spectrums(filtered_spectrums)
        visualize_filtered_spectrum(image, combined_spectrum, combined_filtered_spectrum, title="Combined Spectrum")

        filtered_image = np.stack(filtered_channels, axis=-1)
    else:
        rows, cols = image.shape
        filter_mask = high_pass_directional_filter_mask((rows, cols), angle=45, mask_width=20, cutoff=30)
        spectrum, filtered_spectrum, filtered_dft = process_channel(image, rows, cols, filter_mask)
        visualize_filtered_spectrum(image, spectrum, filtered_spectrum, title="Spectrum")

        filtered_image = np.real(ifft_2d(ifftshift_2d(filtered_dft)))

    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
    if is_color:
        filtered_image_pil = Image.fromarray(filtered_image, mode='RGB')
    else:
        filtered_image_pil = Image.fromarray(filtered_image)
    filtered_image_pil.save("filtered_image.bmp")
    print("Filtered image saved as 'filtered_image.bmp'.")

if __name__ == "__main__":
    main()