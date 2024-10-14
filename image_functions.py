from PIL import Image
import numpy as np

#Subtask B
def doBrightness(param, arr):
    print("Function doBrightness invoked with param: " + str(param))
    arr += int(param)
    arr[arr > 255] = 255  
    arr[arr < 0] = 0 

def doContrast(param, arr):
    print("Function doContrast invoked with param: " + param)
    arr = (arr - 128) * float(param) + 128
    arr[arr > 255] = 255  
    arr[arr < 0] = 0  
    return arr

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
    print("Shrunk image")
    scale_factor = 2**int(param)
    arr = arr[::scale_factor, ::scale_factor]
    return arr

def doEnlarge(param, arr):
    print("Enlarged image")
    scale_factor = 2**int(param)
    arr = np.repeat(np.repeat(arr, scale_factor, axis=0), scale_factor, axis=1)
    return arr

#Subtask E
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
# nie mam pewności czy dobrze