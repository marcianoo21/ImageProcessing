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

def visualize_spectrum(dft, title="Spectrum", channel=None):
    """Visualize the magnitude and phase spectrum of a Fourier transform."""
    magnitude = np.log(np.abs(dft) + 1)  # Apply log for better visualization
    phase = np.angle(dft)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title(f"{title} - Magnitude Spectrum (Channel: {channel})")
    plt.imshow(magnitude, cmap='gray')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title(f"{title} - Phase Spectrum (Channel: {channel})")
    plt.imshow(phase, cmap='gray')
    plt.colorbar()
    plt.savefig(f"{title}_channel_{channel}.png")
    plt.close()

def high_pass_edge_filter(image_path, mask_path):
    # Load the color image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    mask = load_mask(mask_path)  # Load the grayscale mask

    filtered_image = np.zeros_like(image, dtype=np.float32)

    for channel in range(3):
        freq_image = fft2(image[:, :, channel]) 
        filtered_freq = apply_filter(freq_image, mask)  
        visualize_spectrum(filtered_freq, title="Filtered Spectrum", channel=channel)  # Save spectrum visualization

        filtered_channel = np.abs(ifft2(filtered_freq))
        filtered_image[:, :, channel] = filtered_channel

    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image)

    plt.subplot(1, 2, 2)
    plt.title('Filtered Image')
    plt.imshow(filtered_image)

    plt.savefig('output_filtered_image.png')
    plt.close()

high_pass_edge_filter('./images/task4/F5test1.png', './images/task4/F5mask2.png')
