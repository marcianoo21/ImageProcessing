from PIL import Image
import numpy as np
import sys

im = Image.open("lenac.bmp")
im_noised = Image.open("lenac_noised.bmp")

im_noised_resized = im_noised.resize(im.size)

arr = np.array(im)
arr_noised = np.array(im_noised_resized)

if arr.ndim == 1: 
    numColorChannels = 1
    arr = arr.reshape(im.size[1], im.size[0])
else:
    numColorChannels = arr.shape[2]  
    arr = arr.reshape(im.size[1], im.size[0], numColorChannels)

if arr_noised.ndim == 1: 
    numColorChannels = 1
    arr_noised = arr_noised.reshape(im.size[1], im.size[0])
else:
    numColorChannels = arr_noised.shape[2]  
    arr_noised = arr_noised.reshape(im.size[1], im.size[0], numColorChannels)


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
        print("Mean Squared Error: " + str(mse_value))
        return sum / (M * N)
  

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
    elif command == 'pmse':
        pass
    elif command == '--snr':
        pass
    elif command == '--psnr':
        pass
    elif command == 'md':
        pass
    else:
        print("Unknown command: " + command)
        sys.exit()


newIm = Image.fromarray(arr.astype(np.uint8))
newIm.save("result.bmp")