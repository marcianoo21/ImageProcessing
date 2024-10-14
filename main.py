from PIL import Image
import numpy as np
import sys

im = Image.open("lenac.bmp")
im_noised = Image.open("lenac_noised.bmp")

im_noised_resized = im_noised.resize(im.size)

arr = np.array(im)
arr_noised = np.array(im_noised_resized)

if arr.ndim == 2:  # Czarno-biały obraz
    numColorChannels = 1
    arr = arr.reshape(im.size[1], im.size[0])
elif arr.ndim == 3:  # Kolorowy obraz
    numColorChannels = arr.shape[2]
    arr = arr.reshape(im.size[1], im.size[0], numColorChannels)
else:
    raise ValueError("Nieobsługiwany format obrazu")

# Sprawdź, czy obraz z szumem jest czarno-biały czy kolorowy
if arr_noised.ndim == 2:  # Czarno-biały obraz
    numColorChannels = 1
    arr_noised = arr_noised.reshape(im.size[1], im.size[0])
elif arr_noised.ndim == 3:  # Kolorowy obraz
    numColorChannels = arr_noised.shape[2]
    arr_noised = arr_noised.reshape(im.size[1], im.size[0], numColorChannels)
else:
    raise ValueError("Nieobsługiwany format obrazu z szumem")


# print("Image shape (original): " + str(arr.shape))
# print("Image shape (noised resized): " + str(arr_noised.shape))
# print("Number of color channels: " + str(numColorChannels))





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
    scale_factor = int(param)
    arr = arr[::scale_factor, ::scale_factor]
    return arr


def doEnlarge(param, arr):
    print("Enlarged image")
    scale_factor = 2**int(param)
    arr = np.repeat(np.repeat(arr, scale_factor, axis=0), scale_factor, axis=1)
    return arr


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


if len(sys.argv) == 1:
    print("No command line parameters given.\n")
    sys.exit()

command = sys.argv[1]

if len(sys.argv) == 2:
    if command == '--negative':
        arr = doNegative(arr)
    elif command == '--default':
        arr = doDefault(arr)
    elif command == '--vflip':
        arr = doVerticalFlip(arr)
    elif command == '--hflip':
        arr = doHorizontalFlip(arr)
    elif command == '--dflip':
        arr = doDiagonalFlip(arr)
    elif command == '--mse':
        mse_value = mse(arr, arr_noised)
        print("Mean Squared Error: " + str(mse_value))
    elif command == '--pmse':
        pmse_value = pmse(arr, arr_noised)
        print("Peak mean square error: " + str(pmse_value))
    elif command == '--snr':
        snr_value = snr(arr, arr_noised)
        print("Signal to noise ratio: " + str(snr_value))
    elif command == '--psnr':
        psnr_value = psnr(arr, arr_noised)
        print("Peak signal to noise ratio: " + str(psnr_value))
    elif command == '--md':
        md_value = max_diff(arr, arr_noised)
        print("Max difference: " + str(md_value))
    else:
        print("Too few command line parameters given.\n")
        sys.exit()
else:
    param = sys.argv[2]
    if command == '--brightness':
        arr = doBrightness(param, arr)
    elif command == '--contrast':
        arr = doContrast(param, arr)
    elif command == '--shrink':
        arr = doShrink(param, arr)
    elif command == '--enlarge':
        arr = doEnlarge(param, arr)
    elif command == '--snr':
        pass
    elif command == '--psnr':
        pass
    elif command == '--md':
        pass
    else:
        print("Unknown command: " + command)
        sys.exit()


newIm = Image.fromarray(arr.astype(np.uint8))
newIm.save("result.bmp")