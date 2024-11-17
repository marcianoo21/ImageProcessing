# from PIL import Image
# import numpy as np

# def doBrightness(param, arr):
#     try:
#         param = int(param)
#     except ValueError:
#         print(f"Error: Invalid brightness parameter '{param}'. It should be an integer.")
#         return None
    
#     if not (-255 <= param <= 255):
#         print(f"Error: Invalid brightness parameter '{param}'. It should be in the range [-255, 255].")
#         return None

#     print("Function doBrightness invoked with param: " + str(param))
#     arr = arr.astype(np.int16)  # Use a larger dtype to prevent overflow
#     arr += param
#     arr[arr > 255] = 255  
#     arr[arr < 0] = 0    
#     return arr.astype(np.uint8) 

# def doContrast(param, arr):
#     try:
#         param = float(param)
#     except ValueError:
#         print(f"Error: Invalid contrast parameter '{param}'. It should be a number.")
#         return None

#     if not (0 <= param <= 5):
#         print(f"Error: Invalid contrast parameter '{param}'. It should be in the range [0, 5].")
#         return None

#     print("Function doContrast invoked with param: " + str(param))
    
#     # Adjust contrast
#     arr = arr.astype(np.int16)  # Prevent overflow
#     adjusted_arr = (arr - 128) * param + 128
#     adjusted_arr = np.clip(adjusted_arr, 0, 255)  # Clip the values between 0 and 255
#     return adjusted_arr.astype(np.uint8)

# def doNegative(arr):
#     print("Negative action")
#     arr = 255 - arr
#     arr[arr > 255] = 255  
#     arr[arr < 0] = 0  
#     return arr

# def doDefault(arr):
#     print("Default action")
#     im = Image.open("./images/lenac.bmp")
#     arr = np.array(im)
#     return arr

# def doVerticalFlip(arr):
#     print("Vertical flip action")
#     arr = arr[::-1]
#     return arr

# def doHorizontalFlip(arr):
#     print("Horizontal flip action")
#     arr = arr[:, ::-1]
#     return arr

# def doDiagonalFlip(arr):  
#     print("Diagonal flip action")
#     arr = arr[::-1, ::-1]
#     return arr

# def doShrink(param, arr):
#     try:
#         param = float(param)
#     except ValueError:
#         print(f"Error: Invalid shrink parameter '{param}'. It should be a number.")
#         return None

#     if not (1 < param <= 5):
#         print(f"Error: Invalid shrink parameter '{param}'. It should be in the range (1, 5].")
#         return None

#     print(f"Shrunk image by a factor of {param}")
#     scale_factor = param
#     target_shape = (int(arr.shape[0] / scale_factor), int(arr.shape[1] / scale_factor))
#     shrunk_image = np.array([
#         [arr[int(y * scale_factor), int(x * scale_factor)] for x in range(target_shape[1])]
#         for y in range(target_shape[0])
#     ])
#     return shrunk_image

# def doEnlarge(param, arr):
#     try:
#         param = float(param)
#     except ValueError:
#         print(f"Error: Invalid enlarge parameter '{param}'. It should be a number.")
#         return None

#     if not (1 < param <= 5):
#         print(f"Error: Invalid enlarge parameter '{param}'. It should be in the range (1, 5].")
#         return None

#     print(f"Enlarged image by a factor of {param}")
#     scale_factor = param
#     target_shape = (int(arr.shape[0] * scale_factor), int(arr.shape[1] * scale_factor))
#     enlarged_image = np.array([
#         [arr[int(y / scale_factor), int(x / scale_factor)] for x in range(target_shape[1])]
#         for y in range(target_shape[0])
#     ])
#     return enlarged_image

# # def min_filter(arr, kernel_size=2):
# #     output = np.zeros_like(arr, dtype=np.uint8)
# #     padded_arr = np.pad(arr, ((kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2), (0, 0)), mode='edge')

# #     for i in range(arr.shape[0]):
# #         for j in range(arr.shape[1]):
# #             for k in range(arr.shape[2]):  # Loop through each color channel
# #                 output[i, j, k] = np.min(padded_arr[i:i+kernel_size, j:j+kernel_size, k])

# #     return output

# def min_filter(arr, kernel_size=2):
#     if arr.ndim == 2:  # Grayscale image
#         return _apply_min_filter(arr, kernel_size)
#     elif arr.ndim == 3:  # Color image
#         # Apply the filter on each color channel independently
#         filtered_channels = [_apply_min_filter(arr[:, :, ch], kernel_size) for ch in range(arr.shape[2])]
#         return np.stack(filtered_channels, axis=-1)
#     else:
#         raise ValueError("Unsupported image format for min filter")

# def _apply_min_filter(arr, kernel_size):
#     output = np.zeros_like(arr, dtype=np.uint8)
#     padded_arr = np.pad(arr, kernel_size // 2, mode='edge')

#     for i in range(arr.shape[0]):
#         for j in range(arr.shape[1]):
#             region = padded_arr[i:i + kernel_size, j:j + kernel_size]
#             output[i, j] = np.min(region)

#     return output

# # def max_filter(arr, kernel_size=2):
# #     output = np.zeros_like(arr, dtype=np.uint8)
# #     padded_arr = np.pad(arr, ((kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2), (0, 0)), mode='edge')

# #     for i in range(arr.shape[0]):
# #         for j in range(arr.shape[1]):
# #             for k in range(arr.shape[2]):  # Loop through each color channel
# #                 output[i, j, k] = np.max(padded_arr[i:i+kernel_size, j:j+kernel_size, k])

# #     return output

# def max_filter(arr, kernel_size=2):
#     if arr.ndim == 2:  # Grayscale image
#         return _apply_max_filter(arr, kernel_size)
#     elif arr.ndim == 3:  # Color image
#         # Apply the filter on each color channel independently
#         filtered_channels = [_apply_max_filter(arr[:, :, ch], kernel_size) for ch in range(arr.shape[2])]
#         return np.stack(filtered_channels, axis=-1)
#     else:
#         raise ValueError("Unsupported image format for max filter")

# def _apply_max_filter(arr, kernel_size):
#     output = np.zeros_like(arr, dtype=np.uint8)
#     padded_arr = np.pad(arr, kernel_size // 2, mode='edge')

#     for i in range(arr.shape[0]):
#         for j in range(arr.shape[1]):
#             region = padded_arr[i:i + kernel_size, j:j + kernel_size]
#             output[i, j] = np.max(region)

#     return output

# def adaptive_median_filter(arr, max_kernel_size=3):
#     if arr.ndim == 2:  # Grayscale image
#         return _apply_adaptive_median(arr, max_kernel_size)
#     elif arr.ndim == 3:  # Color image
#         # Apply the filter on each color channel independently
#         filtered_channels = [_apply_adaptive_median(arr[:, :, ch], max_kernel_size) for ch in range(arr.shape[2])]
#         return np.stack(filtered_channels, axis=-1)
#     else:
#         raise ValueError("Unsupported image format for adaptive median filter")

# def _apply_adaptive_median(arr, max_kernel_size):
#     output = np.copy(arr)
#     padded_arr = np.pad(arr, max_kernel_size // 2, mode='edge')

#     for i in range(arr.shape[0]):
#         for j in range(arr.shape[1]):
#             for k_size in range(3, max_kernel_size + 1, 2):
#                 region = padded_arr[i:i + k_size, j:j + k_size]
#                 sorted_region = np.sort(region.flatten())
#                 min_val, med_val, max_val = sorted_region[0], np.median(sorted_region), sorted_region[-1]

#                 if med_val > min_val and med_val < max_val:
#                     if arr[i, j] > min_val and arr[i, j] < max_val:
#                         output[i, j] = arr[i, j]
#                     else:
#                         output[i, j] = med_val
#                     break

#     return output

# def mse(arr1, arr2):
#     if len(arr1) != len(arr2) or len(arr1[0]) != len(arr2[0]):
#         print("Images are not the same size.")
#     else:
#         M = len(arr1)
#         N = len(arr1[0])
#         sum = 0
#         for i in range(M):
#             for j in range(N):
#                 sum += (arr1[i][j] - arr2[i][j])**2
#         mse_value = sum / (M * N)

#         return mse_value

# # Można urzyć wbudowanych funkcji do obliczenia MSE
# def pmse(arr1, arr2): 
#     if len(arr1) != len(arr2) or len(arr1[0]) != len(arr2[0]):
#         print("Images are not the same size.")
#     else:
#         M = len(arr1)
#         N = len(arr1[0])
#         sum = 0
#         max_value = np.max(arr1) 
#         for i in range(M):
#             for j in range(N):
#                 sum += ((arr1[i][j] - arr2[i][j])**2 ) / (max_value**2)
#         pmse_value = sum / (M * N)
#     return pmse_value

# # Podobnie tutaj
# def snr(arr1, arr2):
#     if len(arr1) != len(arr2) or len(arr1[0]) != len(arr2[0]):
#         print("Images are not the same size.")
#     else:
#         M = len(arr1)
#         N = len(arr1[0])
#         sum1 = 0
#         sum2 = 0
#         for i in range(M):
#             for j in range(N):
#                 sum1 += arr1[i][j]**2
#                 sum2 += (arr1[i][j] - arr2[i][j])**2
#         if np.all(sum2 == 0): 
#             return float('inf')  

#         snr_value = 10*np.log10(sum1 / sum2)
#     return snr_value

# def psnr(arr1, arr2):
#     max_value = np.max(arr1)
#     mse_value = mse(arr1, arr2)
#     psnr_value = 10*np.log10(max_value**2 / mse_value)

#     return psnr_value 
    
# def max_diff(arr1, arr2):
#     M = len(arr1)
#     N = len(arr1[0])
#     K = len(arr1[0][0])
#     print(M, N, K)
#     pivot = 0
#     for i in range(M):
#         for j in range(N):
#             for k in range(K):
#                 diff = abs(arr1[i][j][k] - arr2[i][j][k])
#                 if diff > pivot:
#                     pivot = diff
#     return pivot


from PIL import Image
import numpy as np
from scipy.ndimage import convolve


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


def laplacian_filter(arr):
    kernel = np.array([[0,  1, 0],
                       [1, -4, 1],
                       [0,  1, 0]])
    if arr.ndim == 2:
        filtered_image = convolve(arr, kernel, mode='reflect')
        return np.clip(filtered_image, 0, 255).astype(np.uint8)
    elif arr.ndim == 3:
        filtered_channels = [convolve(arr[:, :, ch], kernel, mode='reflect') for ch in range(arr.shape[2])]
        filtered_image = np.stack(filtered_channels, axis=-1)
        return np.clip(filtered_image, 0, 255).astype(np.uint8)

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