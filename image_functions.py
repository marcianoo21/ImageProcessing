from PIL import Image
import numpy as np

def doBrightness(param, arr):
    try:
        param = int(param)
    except ValueError:
        print(f"Error: Invalid brightness parameter '{param}'. It should be an integer.")
        return None
    
    if not (-255 <= param <= 255):
        print(f"Error: Invalid brightness parameter '{param}'. It should be in the range [-255, 255].")
        return None

    print("Function doBrightness invoked with param: " + str(param))
    arr = arr.astype(np.int16)  # Use a larger dtype to prevent overflow
    arr += param
    arr[arr > 255] = 255  
    arr[arr < 0] = 0    
    return arr.astype(np.uint8) 

def doContrast(param, arr):
    try:
        param = float(param)
    except ValueError:
        print(f"Error: Invalid contrast parameter '{param}'. It should be a number.")
        return None

    if not (0 <= param <= 5): 
        print(f"Error: Invalid contrast parameter '{param}'. It should be in the range [0, 5].")
        return None

    print("Function doContrast invoked with param: " + str(param))
    
    # Adjust contrast
    arr = arr * param
    arr = arr.astype(np.int16)  # Prevent overflow
    arr[arr > 255] = 255  
    arr[arr < 0] = 0  
    return arr.astype(np.uint8)

def doNegative(arr):
    print("Negative action")
    arr = 255 - arr
    arr[arr > 255] = 255  
    arr[arr < 0] = 0  
    return arr

def doDefault(arr):
    print("Default action")
    im = Image.open("lenac.bmp")
    arr = np.array(im)
    return arr

def doVerticalFlip(arr):
    print("Vertical flip action")
    arr = arr[::-1]
    return arr

def doHorizontalFlip(arr):
    print("Horizontal flip action")
    arr = arr[:, ::-1]
    return arr

def doDiagonalFlip(arr):  
    print("Diagonal flip action")
    arr = arr[::-1, ::-1]
    return arr

def doShrink(param, arr):
    try:
        param = float(param)
    except ValueError:
        print(f"Error: Invalid shrink parameter '{param}'. It should be a number.")
        return None

    if not (1 < param <= 5):
        print(f"Error: Invalid shrink parameter '{param}'. It should be in the range (1, 5].")
        return None

    print(f"Shrunk image by a factor of {param}")
    scale_factor = param
    target_shape = (int(arr.shape[0] / scale_factor), int(arr.shape[1] / scale_factor))
    shrunk_image = np.array([
        [arr[int(y * scale_factor), int(x * scale_factor)] for x in range(target_shape[1])]
        for y in range(target_shape[0])
    ])
    return shrunk_image

def doEnlarge(param, arr):
    try:
        param = float(param)
    except ValueError:
        print(f"Error: Invalid enlarge parameter '{param}'. It should be a number.")
        return None

    if not (1 < param <= 5):
        print(f"Error: Invalid enlarge parameter '{param}'. It should be in the range (1, 5].")
        return None

    print(f"Enlarged image by a factor of {param}")
    scale_factor = param
    target_shape = (int(arr.shape[0] * scale_factor), int(arr.shape[1] * scale_factor))
    enlarged_image = np.array([
        [arr[int(y / scale_factor), int(x / scale_factor)] for x in range(target_shape[1])]
        for y in range(target_shape[0])
    ])
    return enlarged_image

def min_filter(arr, kernel_size=3):
    output = np.zeros_like(arr)
    padded_arr = np.pad(arr, kernel_size // 2, mode='edge')

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            output[i, j] = np.min(padded_arr[i:i+kernel_size, j:j+kernel_size])

    return output

def max_filter(arr, kernel_size=3):
    output = np.zeros_like(arr)
    padded_arr = np.pad(arr, kernel_size // 2, mode='edge')

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            output[i, j] = np.max(padded_arr[i:i+kernel_size, j:j+kernel_size])

    return output

def adaptive_median_filter(arr, max_kernel_size=7):
    if arr.ndim == 2:  # Grayscale image
        return _apply_adaptive_median(arr, max_kernel_size)
    elif arr.ndim == 3:  # Color image
        # Apply the filter on each color channel independently
        filtered_channels = [ _apply_adaptive_median(arr[:, :, ch], max_kernel_size) for ch in range(arr.shape[2])]
        return np.stack(filtered_channels, axis=-1)
    else:
        raise ValueError("Unsupported image format for adaptive median filter")

def _apply_adaptive_median(arr, max_kernel_size):
    output = np.copy(arr)
    padded_arr = np.pad(arr, max_kernel_size // 2, mode='edge')

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            for k_size in range(3, max_kernel_size+1, 2):
                region = padded_arr[i:i+k_size, j:j+k_size]
                sorted_region = np.sort(region.flatten())
                min_val, med_val, max_val = sorted_region[0], np.median(sorted_region), sorted_region[-1]

                if med_val > min_val and med_val < max_val:
                    if arr[i, j] > min_val and arr[i, j] < max_val:
                        output[i, j] = arr[i, j]
                    else:
                        output[i, j] = med_val
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