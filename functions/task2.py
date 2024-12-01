from PIL import Image
import numpy as np
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import cv2
import os

# def universal_laplacian_filter(arr, kernel):
#     if arr.ndim == 2:  # Grayscale image
#         filtered_image = convolve(arr, kernel, mode='reflect')
#         return np.clip(filtered_image, 0, 255).astype(np.uint8)
#     elif arr.ndim == 3:  # Color image
#         filtered_channels = [convolve(arr[:, :, ch], kernel, mode='reflect') for ch in range(arr.shape[2])]
#         filtered_image = np.stack(filtered_channels, axis=-1)
#         return np.clip(filtered_image, 0, 255).astype(np.uint8)
    
def universal_laplacian_filter(image, mask):
    def normalize(channel):
        min_val = np.min(channel)
        max_val = np.max(channel)
        if max_val - min_val == 0:
            return np.zeros_like(channel, dtype=np.uint8)
        normalized_channel = 255 * (channel - min_val) / (max_val - min_val)
        return normalized_channel.astype(np.uint8)

    def apply_filter_to_channel(channel, mask):
        k = mask.shape[0] // 2  
        
        padded_channel = cv2.copyMakeBorder(channel, k, k, k, k, cv2.BORDER_REPLICATE)
        
        filtered_channel = np.zeros_like(channel, dtype=np.float32)
        
        for i in range(channel.shape[0]):
            for j in range(channel.shape[1]):
                roi = padded_channel[i:i + mask.shape[0], j:j + mask.shape[1]]
                filtered_channel[i, j] = np.sum(roi * mask)
        
        return normalize(filtered_channel)

    if len(image.shape) == 3:  
        channels = cv2.split(image)
        filtered_channels = []
        for channel in channels:
            filtered_channel = apply_filter_to_channel(channel, mask)
            filtered_channels.append(filtered_channel)
        return cv2.merge(filtered_channels)
    else:  
        return apply_filter_to_channel(image, mask)


def optimized_laplacian_filter(arr):
    laplacian_kernel = np.array([[1, -2, 1],
                                 [-2, 4, -2],
                                 [1, -2, 1]])
    
    if arr.ndim == 2:  
        filtered_image = convolve(arr, laplacian_kernel, mode='nearest')
        edge_image = np.abs(filtered_image)  
        edge_image = np.clip(edge_image, 0, 255)  
        return edge_image.astype(np.uint8)

  
    elif arr.ndim == 3:  
        filtered_image = np.zeros_like(arr)
        for ch in range(arr.shape[2]):
            filtered_image[:, :, ch] = convolve(arr[:, :, ch], laplacian_kernel, mode='nearest')
        
        edge_image = np.abs(filtered_image)
        edge_image = np.clip(edge_image, 0, 255)  
        return edge_image.astype(np.uint8)

    else:
        raise ValueError("Input array must be a 2D (grayscale) or 3D (color) image.")

def roberts_operator_ii(image):
    if len(image.shape) == 3:  
        channels = cv2.split(image)
        processed_channels = []
        for channel in channels:
            processed_channel = roberts_operator_ii_single_channel(channel)
            processed_channels.append(processed_channel)
        return cv2.merge(processed_channels)  
    else: 
        return roberts_operator_ii_single_channel(image)

def roberts_operator_ii_single_channel(image):
    height, width = image.shape
    
    filtered_image = np.zeros_like(image, dtype=np.float32)
    
    for n in range(height - 1):  # Exclude the last row
        for m in range(width - 1):  # Exclude the last column
            diff1 = abs(float(image[n, m]) - float(image[n + 1, m + 1]))
            diff2 = abs(float(image[n, m + 1]) - float(image[n + 1, m]))
            
            filtered_image[n, m] = diff1 + diff2
    
    filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX)
    return filtered_image.astype(np.uint8)

def mean(histogram, total_pixels):
    return np.sum(histogram * np.arange(len(histogram))) / total_pixels

def variance(histogram, total_pixels, mean):
    return np.sum(((np.arange(len(histogram)) - mean) ** 2) * histogram) / total_pixels

def std_dev(variance):
    return np.sqrt(variance)

def variation_coeff_1(std_dev, mean):
    return std_dev / mean

def asymmetry_coeff(histogram, total_pixels, mean, std_dev):
    return np.sum(((np.arange(len(histogram)) - mean) ** 3) * histogram) / (std_dev ** 3 * total_pixels)

def flattening_coeff(histogram, total_pixels, mean, std_dev):
    return (np.sum(((np.arange(len(histogram)) - mean) ** 4) * histogram) / (std_dev ** 4 * total_pixels)) - 3

def variation_coeff_2(histogram, total_pixels):
    return np.sum((histogram / total_pixels) ** 2)

def entropy(histogram, total_pixels):
    return -np.sum((histogram / total_pixels) * np.log2(histogram / total_pixels + 1e-10))

def create_histogram(arr, output_dir="histograms"):
    if arr is None:
        raise ValueError("Input image array is None. Ensure the image is loaded correctly.")

    os.makedirs(output_dir, exist_ok=True)

    if len(arr.shape) == 2: 
        histogram, bins = np.histogram(arr, bins=256, range=[0, 256])
        plt.figure()
        plt.plot(histogram, color='black')
        plt.title("Grayscale Histogram")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        output_path = os.path.join(output_dir, "histogram_grayscale.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Grayscale histogram saved at {output_path}")
    elif len(arr.shape) == 3: 
        channels = cv2.split(arr)
        colors = ['blue', 'green', 'red']
        for i, (channel, color) in enumerate(zip(channels, colors)):
            histogram, bins = np.histogram(channel, bins=256, range=[0, 256])
            plt.figure()
            plt.plot(histogram, color=color)
            plt.title(f"{color.capitalize()} Channel Histogram")
            plt.xlabel("Pixel Intensity")
            plt.ylabel("Frequency")
            output_path = os.path.join(output_dir, f"histogram_{color}.png")
            plt.savefig(output_path)
            plt.close()
            print(f"{color.capitalize()} channel histogram saved at {output_path}")
    else:
        raise ValueError("Unsupported image format or corrupted image.")

    return arr

def apply_exponential_transformation(channel, g_min, g_max):
    N = channel.size
    histogram, _ = np.histogram(channel, bins=256, range=[0, 256])
    cdf = np.cumsum(histogram) / N  

    epsilon = 1e-10
    cdf_max = cdf[-1] 
    alpha = (g_min - g_max) / np.log(np.maximum(epsilon, 1 - cdf_max))

    transform = lambda f: g_min - alpha * np.log(1 - cdf[f]) if cdf[f] < 1 else g_max
    new_channel = np.zeros_like(channel, dtype=np.float32)
    for f in range(256):
        new_channel[channel == f] = transform(f)

    new_channel = np.clip(new_channel, g_min, g_max)
    return new_channel

def exponential_density_function(image, g_min, g_max):
    if len(image.shape) == 2: 
        return apply_exponential_transformation(image, g_min, g_max)

    elif len(image.shape) == 3: 
        new_image = np.zeros_like(image, dtype=np.float32)
        for c in range(image.shape[2]): 
            channel = image[:, :, c]
            new_image[:, :, c] = apply_exponential_transformation(channel, g_min, g_max)
        
        return new_image.astype(np.uint8)

    else:
        raise ValueError("Unsupported image format. The image should be 2D (grayscale) or 3D (color).")