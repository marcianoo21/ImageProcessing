from PIL import Image
import numpy as np

# Basic Image Operations
def do_brightness(param, arr):
    param = int(param)
    arr = arr.astype(np.int16)
    arr += param
    arr[arr > 255] = 255
    arr[arr < 0] = 0
    return arr.astype(np.uint8)

def do_contrast(param, arr):
    param = float(param)
    arr = arr.astype(np.int16)
    adjusted_arr = (arr - 128) * param + 128
    adjusted_arr = np.clip(adjusted_arr, 0, 255)
    return adjusted_arr.astype(np.uint8)

def do_negative(arr):
    return 255 - arr

def do_default():
    im = Image.open("./images/lenac.bmp")
    return np.array(im)

def do_vertical_flip(arr):
    return arr[::-1]

def do_horizontal_flip(arr):
    return arr[:, ::-1]

def do_diagonal_flip(arr):
    return arr[::-1, ::-1]

def do_shrink(arr, param):
    param = float(param)
    target_shape = (int(arr.shape[0] / param), int(arr.shape[1] / param))
    shrunk_image = np.array([
        [arr[int(y * param), int(x * param)] for x in range(target_shape[1])]
        for y in range(target_shape[0])
    ])
    return shrunk_image

def do_enlarge(param, arr):
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