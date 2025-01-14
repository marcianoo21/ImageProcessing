import numpy as np
import cv2
from matplotlib import pyplot as plt

def fft2(image):
    return np.fft.fftshift(np.fft.fft2(image))

def ifft2(freq_image):
    return np.fft.ifft2(np.fft.ifftshift(freq_image))

def apply_filter(freq_image, mask):
    return freq_image * mask

def load_mask(mask_path):
    return cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0

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


def high_pass_edge_filter(image_path, mask_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = load_mask(mask_path) 

    freq_image = fft2(image)
    filtered_freq = apply_filter(freq_image, mask)
    visualize_spectrum(filtered_freq, title="Filtered Spectrum")
    filtered_image = np.abs(ifft2(filtered_freq))
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title(f'Filtered Image')
    plt.imshow(filtered_image, cmap='gray')
    
    plt.savefig('output.png')
    plt.close()
# Example usage
high_pass_edge_filter('./images/task4/F5test1.png', './images/task4/F5mask2.png')