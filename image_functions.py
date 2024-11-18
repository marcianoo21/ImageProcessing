from PIL import Image
import numpy as np
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import cv2
import os


# Basic Image Operations
def doBrightness(param, arr):
    param = int(param)
    arr = arr.astype(np.int16)
    arr += param
    arr[arr > 255] = 255
    arr[arr < 0] = 0
    return arr.astype(np.uint8)


def doContrast(param, arr):
    param = float(param)
    arr = arr.astype(np.int16)
    adjusted_arr = (arr - 128) * param + 128
    adjusted_arr = np.clip(adjusted_arr, 0, 255)
    return adjusted_arr.astype(np.uint8)


def doNegative(arr):
    return 255 - arr


def doDefault():
    im = Image.open("./images/lenac.bmp")
    return np.array(im)


def doVerticalFlip(arr):
    return arr[::-1]


def doHorizontalFlip(arr):
    return arr[:, ::-1]


def doDiagonalFlip(arr):
    return arr[::-1, ::-1]


def doShrink(param, arr):
    param = float(param)
    target_shape = (int(arr.shape[0] / param), int(arr.shape[1] / param))
    shrunk_image = np.array([
        [arr[int(y * param), int(x * param)] for x in range(target_shape[1])]
        for y in range(target_shape[0])
    ])
    return shrunk_image


def doEnlarge(param, arr):
    param = float(param)
    target_shape = (int(arr.shape[0] * param), int(arr.shape[1] * param))
    enlarged_image = np.array([
        [arr[int(y / param), int(x / param)] for x in range(target_shape[1])]
        for y in range(target_shape[0])
    ])
    return enlarged_image


# Filters
def min_filter(arr, kernel_size=2):
    return _apply_filter(arr, kernel_size, np.min)


def max_filter(arr, kernel_size=2):
    return _apply_filter(arr, kernel_size, np.max)


def _apply_filter(arr, kernel_size, operation):
    if arr.ndim == 2:
        return _apply_filter_single(arr, kernel_size, operation)
    elif arr.ndim == 3:
        filtered_channels = [_apply_filter_single(arr[:, :, ch], kernel_size, operation)
                             for ch in range(arr.shape[2])]
        return np.stack(filtered_channels, axis=-1)
    else:
        raise ValueError("Unsupported image format")


def _apply_filter_single(arr, kernel_size, operation):
    output = np.zeros_like(arr, dtype=np.uint8)
    padded_arr = np.pad(arr, kernel_size // 2, mode='edge')
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            region = padded_arr[i:i + kernel_size, j:j + kernel_size]
            output[i, j] = operation(region)
    return output


def adaptive_median_filter(arr, max_kernel_size=3):
    if arr.ndim == 2:
        return _adaptive_filter(arr, max_kernel_size)
    elif arr.ndim == 3:
        filtered_channels = [_adaptive_filter(arr[:, :, ch], max_kernel_size)
                             for ch in range(arr.shape[2])]
        return np.stack(filtered_channels, axis=-1)
    else:
        raise ValueError("Unsupported image format")


def _adaptive_filter(arr, max_kernel_size):
    output = np.copy(arr)
    padded_arr = np.pad(arr, max_kernel_size // 2, mode='edge')
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            for k_size in range(3, max_kernel_size + 1, 2):
                region = padded_arr[i:i + k_size, j:j + k_size]
                sorted_region = np.sort(region.flatten())
                min_val, med_val, max_val = sorted_region[0], np.median(sorted_region), sorted_region[-1]
                if med_val > min_val and med_val < max_val:
                    output[i, j] = med_val if not (min_val < arr[i, j] < max_val) else arr[i, j]
                    break
    return output


def mse(arr1, arr2):
    if len(arr1) != len(arr2) or len(arr1[0]) != len(arr2[0]):
        print("Images are not the same size.")
    else:
        M = len(arr1)
        N = len(arr1[0])
        sum = 0
        for i in range(M):
            for j in range(N):
                sum += (arr1[i][j] - arr2[i][j])**2
        mse_value = sum / (M * N)

        return mse_value

# Można urzyć wbudowanych funkcji do obliczenia MSE
def pmse(arr1, arr2): 
    if len(arr1) != len(arr2) or len(arr1[0]) != len(arr2[0]):
        print("Images are not the same size.")
    else:
        M = len(arr1)
        N = len(arr1[0])
        sum = 0
        max_value = np.max(arr1) 
        for i in range(M):
            for j in range(N):
                sum += ((arr1[i][j] - arr2[i][j])**2 ) / (max_value**2)
        pmse_value = sum / (M * N)
    return pmse_value

# Podobnie tutaj
def snr(arr1, arr2):
    if len(arr1) != len(arr2) or len(arr1[0]) != len(arr2[0]):
        print("Images are not the same size.")
    else:
        M = len(arr1)
        N = len(arr1[0])
        sum1 = 0
        sum2 = 0
        for i in range(M):
            for j in range(N):
                sum1 += arr1[i][j]**2
                sum2 += (arr1[i][j] - arr2[i][j])**2
        if np.all(sum2 == 0): 
            return float('inf')  

        snr_value = 10*np.log10(sum1 / sum2)
    return snr_value

def psnr(arr1, arr2):
    max_value = np.max(arr1)
    mse_value = mse(arr1, arr2)
    psnr_value = 10*np.log10(max_value**2 / mse_value)

    return psnr_value 
    
def max_diff(arr1, arr2):
    M = len(arr1)
    N = len(arr1[0])
    K = len(arr1[0][0])
    print(M, N, K)
    pivot = 0
    for i in range(M):
        for j in range(N):
            for k in range(K):
                diff = abs(arr1[i][j][k] - arr2[i][j][k])
                if diff > pivot:
                    pivot = diff
    return pivot

def universal_laplacian_filter(arr, kernel):
    if arr.ndim == 2:  # Grayscale image
        filtered_image = convolve(arr, kernel, mode='reflect')
        return np.clip(filtered_image, 0, 255).astype(np.uint8)
    elif arr.ndim == 3:  # Color image
        filtered_channels = [convolve(arr[:, :, ch], kernel, mode='reflect') for ch in range(arr.shape[2])]
        filtered_image = np.stack(filtered_channels, axis=-1)
        return np.clip(filtered_image, 0, 255).astype(np.uint8)
    
# def universal_laplacian_filter(image, mask):
#     def normalize(channel):
#         min_val = np.min(channel)
#         max_val = np.max(channel)
#         if max_val - min_val == 0:
#             return np.zeros_like(channel, dtype=np.uint8)
#         normalized_channel = 255 * (channel - min_val) / (max_val - min_val)
#         return normalized_channel.astype(np.uint8)

#     def apply_filter_to_channel(channel, mask):
#         k = mask.shape[0] // 2  
        
#         padded_channel = cv2.copyMakeBorder(channel, k, k, k, k, cv2.BORDER_REPLICATE)
        
#         filtered_channel = np.zeros_like(channel, dtype=np.float32)
        
#         for i in range(channel.shape[0]):
#             for j in range(channel.shape[1]):
#                 roi = padded_channel[i:i + mask.shape[0], j:j + mask.shape[1]]
#                 filtered_channel[i, j] = np.sum(roi * mask)
        
#         return normalize(filtered_channel)

#     if len(image.shape) == 3:  
#         channels = cv2.split(image)
#         filtered_channels = []
#         for channel in channels:
#             filtered_channel = apply_filter_to_channel(channel, mask)
#             filtered_channels.append(filtered_channel)
#         return cv2.merge(filtered_channels)
#     else:  
#         return apply_filter_to_channel(image, mask)


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

def exponential_density_function(image, g_min = 0, g_max = 255):
    if len(image.shape) == 3:
        raise ValueError("Exponential density function is implemented for grayscale images only.")

    N = image.size
    histogram, _ = np.histogram(image, bins=256, range=[0, 256])
    cdf = np.cumsum(histogram) / N  
    alpha = (g_max - g_min) / np.log(1 + cdf.max())  

    transform = lambda f: g_min - alpha * np.log(1 - cdf[f])
    new_image = np.zeros_like(image, dtype=np.float32)
    for f in range(256):
        new_image[image == f] = transform(f)

    new_image = np.clip(new_image, g_min, g_max)
    return new_image.astype(np.uint8)