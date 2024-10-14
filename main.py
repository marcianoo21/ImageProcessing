from PIL import Image
import numpy as np
import sys
from image_functions import (doBrightness, doContrast, doNegative, 
                             doDefault, doVerticalFlip, doHorizontalFlip, 
                             doDiagonalFlip, doShrink, doEnlarge)

def print_help():
    help_text = """
    Available Commands:
    
    --help              : Show this help message and exit.
    
    --negative          : Apply a negative filter to the image.
    
    --default           : Reset the image to the original state.
    
    --vflip             : Apply vertical flip to the image.
    
    --hflip             : Apply horizontal flip to the image.
    
    --dflip             : Apply diagonal flip to the image.
    
    --brightness <val>  : Adjust brightness by the specified value. 
                          Example: --brightness 50
    
    --contrast <val>    : Adjust contrast by the specified factor.
                          Example: --contrast 1.2
    
    --shrink <val>      : Shrink the image by a factor of 2^<val>.
                          Example: --shrink 1 (shrinks the image by a factor of 2)
    
    --enlarge <val>     : Enlarge the image by a factor of 2^<val>.
                          Example: --enlarge 1 (enlarges the image by a factor of 2)
    
    --mse               : Calculate Mean Squared Error between original and noised image.
    
    --pmse              : Calculate Peak Mean Square Error.
    
    --snr               : Calculate Signal-to-Noise Ratio.
    
    --psnr              : Calculate Peak Signal-to-Noise Ratio.
    
    --md                : Calculate maximum difference between the original and noised image.
    """
    print(help_text)

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

if len(sys.argv) == 1:
    print("No command line parameters given.\n")
    sys.exit()

command = sys.argv[1]

if len(sys.argv) == 2:
    match command:
        case '--help':
            print_help()
            sys.exit()
        case '--negative':
            arr = doNegative(arr)
        case '--default':
            arr = doDefault(arr)
        case '--vflip':
            arr = doVerticalFlip(arr)
        case '--hflip':
            arr = doHorizontalFlip(arr)
        case '--dflip':
            arr = doDiagonalFlip(arr)
        case '--mse':
            mse_value = mse(arr, arr_noised)
            print("Mean Squared Error: " + str(mse_value))
        case '--pmse':
            pmse_value = pmse(arr, arr_noised)
            print("Peak mean square error: " + str(pmse_value))
        case '--snr':
            snr_value = snr(arr, arr_noised)
            print("Signal to noise ratio: " + str(snr_value))
        case '--psnr':
            psnr_value = psnr(arr, arr_noised)
            print("Peak signal to noise ratio: " + str(psnr_value))
        case '--md':
            md_value = max_diff(arr, arr_noised)
            print("Max difference: " + str(md_value))
        case _:
            print("Unknown command: " + command)
            sys.exit()
else:
    param = sys.argv[2]
    match command:
        case '--brightness':
            arr = doBrightness(param, arr)
        case '--contrast':
            arr = doContrast(param, arr)
        case '--shrink':
            arr = doShrink(param, arr)
        case '--enlarge':
            arr = doEnlarge(param, arr)
        case _:
            print("Unknown command: " + command)
            sys.exit()

newIm = Image.fromarray(arr.astype(np.uint8))
newIm.save("result.bmp")
