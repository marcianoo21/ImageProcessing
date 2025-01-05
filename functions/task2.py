import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt

def filter2D(arr, kernel):
    arr = arr.astype(np.float32)
    kernel = kernel.astype(np.float32)
    height, width = arr.shape
    k_height, k_width = kernel.shape
    pad_y, pad_x = k_height // 2, k_width // 2 # padding for kernel to be applied to every pixel
    filtered_image = np.zeros_like(arr)
    padded_image = np.pad(arr, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant', constant_values=0)

    for i in range(height):
        for j in range(width):
            window = padded_image[i:i + k_height, j:j + k_width]
            filtered_value = np.sum(window * kernel)
            filtered_image[i, j] = filtered_value

    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
    
    return filtered_image


def universal_laplacian_filter(arr, kernel):
    if arr.ndim == 2:
        filtered_image = filter2D(arr, kernel)
        return np.clip(filtered_image, 0, 255).astype(np.uint8)
    elif arr.ndim == 3:
        filtered_channels = [filter2D(arr[:, :, ch], kernel) for ch in range(arr.shape[2])]
        filtered_image = np.stack(filtered_channels, axis=-1)
        return np.clip(filtered_image, 0, 255).astype(np.uint8)
    else:
        raise ValueError("Input array must be a 2D (grayscale) or 3D (color)") 
         

def optimized_laplacian_filter(arr):
    if arr.ndim == 2:
        return laplacian_convolution(arr)
    
    elif arr.ndim == 3:
        filtered_image = np.zeros_like(arr, dtype=np.uint8)
        for ch in range(arr.shape[2]):
            filtered_image[:, :, ch] = laplacian_convolution(arr[:, :, ch])
        return filtered_image
    
    else:
        raise ValueError("Input array must be a 2D (grayscale) or 3D (color) image.")

def laplacian_convolution(image):
    height, width = image.shape

    filtered_image = np.zeros_like(image, dtype=np.float32)

    padded_image = np.pad(image, ((1, 1), (1, 1)), mode='constant', constant_values=0).astype(np.float32)

    for i in range(1, height + 1):
        for j in range(1, width + 1):
            center = 8 * padded_image[i, j]

            neighbors = (
                padded_image[i - 1, j - 1] + padded_image[i - 1, j] + padded_image[i - 1, j + 1] +
                padded_image[i, j - 1] + padded_image[i, j + 1] +
                padded_image[i + 1, j - 1] + padded_image[i + 1, j] + padded_image[i + 1, j + 1]
            )
            
            filtered_image[i - 1, j - 1] = center - neighbors

    edge_image = np.clip(np.abs(filtered_image), 0, 255).astype(np.uint8)
    return edge_image

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
    
    for n in range(height - 1):  
        for m in range(width - 1): 
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

def create_histogram(arr, output_dir="histograms", channels=None):
    if arr is None:
        raise ValueError("Input image array is None. Ensure the image is loaded correctly.")
    
    # Ensure the directory exists and is empty
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if len(arr.shape) == 2:  # Grayscale image
        histogram, bins = np.histogram(arr, bins=256, range=[0, 256])
        plt.figure()
        plt.plot(histogram, color='black', alpha=0.7)
        plt.title("Grayscale Histogram")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        output_path = os.path.join(output_dir, "histogram_grayscale.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Grayscale histogram saved at {output_path}")

    elif len(arr.shape) == 3:  # Color image
        all_channels = {'blue': 0, 'green': 1, 'red': 2}
        colors = ['blue', 'green', 'red']
        channel_indices = []

        if channels is None:  
            channel_indices = list(all_channels.values())
            selected_colors = colors
        else:  
            if not isinstance(channels, (list, tuple)):
                raise ValueError("Channels must be a list or tuple of 'red', 'green', and/or 'blue'.")
            selected_channels = [ch.lower() for ch in channels]
            if not all(ch in all_channels for ch in selected_channels):
                raise ValueError("Invalid channel specified. Choose from 'red', 'green', or 'blue'.")
            channel_indices = [all_channels[ch] for ch in selected_channels]
            selected_colors = [colors[i] for i in channel_indices]

        plt.figure()
        for idx, color in zip(channel_indices, selected_colors):
            histogram, bins = np.histogram(cv2.split(arr)[idx], bins=256, range=[0, 256])
            plt.plot(histogram, color=color, alpha=0.7, label=f"{color.capitalize()} Channel")
        
        plt.title("Selected Channels Histogram")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.legend()
        output_path = os.path.join(output_dir, "histogram_selected_channels.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Histogram for selected channels saved at {output_path}")

    else:
        raise ValueError("Unsupported image format or corrupted image.")

    return arr

def uniform_pdf(channel, g_min, g_max):
    histogram, _ = np.histogram(channel, bins=256, range=[0, 256])
    N = channel.size

    cdf = np.cumsum(histogram)

    transform = lambda f: g_min + (g_max - g_min) * (1 / N) * cdf[f]

    transformed_channel = np.zeros_like(channel, dtype=np.float32)
    for f in range(256):
        transformed_channel[channel == f] = transform(f)

    transformed_channel = np.clip(transformed_channel, g_min, g_max)

    return transformed_channel.astype(np.uint8)


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

def exponential_density_function(image, g_min, g_max, mode="default", ref_channel="green"):

    if len(image.shape) == 2:  
        return apply_exponential_transformation(image, g_min, g_max)

    elif len(image.shape) == 3: 
        all_channels = {'blue': 0, 'green': 1, 'red': 2}
        if ref_channel not in all_channels:
            raise ValueError(f"Invalid reference channel: {ref_channel}. Choose from 'blue', 'green', or 'red'.")
        
        ref_idx = all_channels[ref_channel]
        ref = image[:, :, ref_idx]

        if mode == "default":
            new_image = np.zeros_like(image, dtype=np.float32)
            for c in range(image.shape[2]): 
                new_image[:, :, c] = apply_exponential_transformation(image[:, :, c], g_min, g_max)
            return new_image.astype(np.uint8)

        elif mode == "difference":
            differences = [ref - image[:, :, i] for i in range(image.shape[2]) if i != ref_idx]
            transformed_ref = apply_exponential_transformation(ref, g_min, g_max)
            new_image = np.zeros_like(image, dtype=np.float32)
            new_image[:, :, ref_idx] = transformed_ref

            idx = 0
            for i in range(image.shape[2]):
                if i != ref_idx:
                    new_channel = transformed_ref + differences[idx]
                    new_channel = np.clip(new_channel, 0, 255)
                    new_image[:, :, i] = new_channel
                    idx += 1

            return new_image.astype(np.uint8)

        elif mode == "ratio":
            ratios = [image[:, :, i] / (ref + 1e-10) for i in range(image.shape[2]) if i != ref_idx]
            transformed_ref = apply_exponential_transformation(ref, g_min, g_max)
            new_image = np.zeros_like(image, dtype=np.float32)
            new_image[:, :, ref_idx] = transformed_ref

            idx = 0
            for i in range(image.shape[2]):
                if i != ref_idx:
                    new_channel = transformed_ref * ratios[idx]
                    new_channel = np.clip(new_channel, 0, 255)
                    new_image[:, :, i] = new_channel
                    idx += 1

            return new_image.astype(np.uint8)

        else:
            raise ValueError(f"Unsupported mode: {mode}. Choose from 'default', 'difference', or 'ratio'.")
    else:
        raise ValueError("Unsupported image format. The image should be 2D (grayscale) or 3D (color).")
    
def apply_hiperbolic_transformation(channel, g_min, g_max):
    N = channel.size
    histogram, _ = np.histogram(channel, bins=256, range=[0, 256])
    cdf = np.cumsum(histogram) / N  

    transform = lambda f: g_min * (g_max / g_min) ** cdf[f] if cdf[f] < 1 else g_max
    new_channel = np.zeros_like(channel, dtype=np.float32)
    for f in range(256):
        new_channel[channel == f] = transform(f)

    new_channel = np.clip(new_channel, g_min, g_max)
    return new_channel

def hiperbolic_density_function(image, g_min, g_max):
    if g_min >= 1:
        if len(image.shape) == 2:  
            return apply_hiperbolic_transformation(image, g_min, g_max)

        elif len(image.shape) == 3:         
            new_image = np.zeros_like(image, dtype=np.float32)
            for c in range(image.shape[2]): 
                new_image[:, :, c] = apply_hiperbolic_transformation(image[:, :, c], g_min, g_max)
            return new_image.astype(np.uint8)

        else:
            raise ValueError("Unsupported image format. The image should be 2D (grayscale) or 3D (color).")
    else:
        raise ValueError("Minimal value for g_min is 1")
