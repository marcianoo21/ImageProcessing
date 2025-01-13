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

def fft_2d(image):
    """Compute the 2D Fast Fourier Transform using np.fft.fft2."""
    return np.fft.fft2(image)

def ifft_2d(dft):
    """Compute the 2D Inverse Fast Fourier Transform using np.fft.ifft2."""
    return np.fft.ifft2(dft)

def fft_2d(image):
    """Compute the 2D Fast Fourier Transform manually."""
    N, M = image.shape
    # FFT in rows
    row_fft = np.zeros((N, M), dtype=complex)
    for i in range(N):
        row_fft[i, :] = np.fft.fft(image[i, :])
    # FFT in columns
    col_fft = np.zeros((N, M), dtype=complex)
    for j in range(M):
        col_fft[:, j] = np.fft.fft(row_fft[:, j])
    return col_fft

def ifft_2d(dft):
    """Compute the 2D Inverse Fast Fourier Transform manually."""
    N, M = dft.shape
    # IFFT in columns
    col_ifft = np.zeros((N, M), dtype=complex)
    for j in range(M):
        col_ifft[:, j] = np.fft.ifft(dft[:, j])
    # IFFT in rows
    row_ifft = np.zeros((N, M), dtype=complex)
    for i in range(N):
        row_ifft[i, :] = np.fft.ifft(col_ifft[i, :])
    return row_ifft

def visualize_spectrum(dft, title="Spectrum"):
    """Visualize the magnitude and phase spectrum of a Fourier transform."""
    magnitude = np.log(np.abs(dft) + 1)  # Apply log for better visualization
    phase = np.angle(dft)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title(f"{title} - Magnitude Spectrum")
    plt.imshow(magnitude, cmap='gray')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title(f"{title} - Phase Spectrum")
    plt.imshow(phase, cmap='gray')
    plt.colorbar()
    plt.savefig(f"{title}_spectrum.png")
    plt.close()

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

def apply_log_scaling(image):
    """Apply logarithmic scaling to enhance visibility of small values."""
    return np.log(np.abs(image) + 1)

if __name__ == "__main__":
    image = np.array(Image.open('./images/lena_shrink_4.bmp'))  
    start_time = time.time()
    dft = dft_2d(image)

    # visualize_spectrum(dft, title="Before Filtering")

    # print(f"shape: {dft.shape}")
    # f_filter = create_low_pass_filter(dft.shape, cutoff=362) #362 and 363
    # filtered_dft = apply_filter(dft, f_filter)

    # visualize_spectrum(filtered_dft, title="After Filtering")

    # print(f"Filtered DFT Stats - min: {np.min(filtered_dft)}, max: {np.max(filtered_dft)}, mean: {np.mean(filtered_dft)}")

    filtered_image = np.real(idft_2d(dft))  
    
    end_time = time.time()

    print(f"Execution time: {end_time - start_time:.4f} seconds")

    print(f"Filtered Image Stats - min: {np.min(filtered_image)}, max: {np.max(filtered_image)}, mean: {np.mean(filtered_image)}")

    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)

    Image.fromarray(filtered_image).save("filtered_image.bmp")

    print("Filtered image saved as 'filtered_image.bmp'.")