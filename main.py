from PIL import Image
import numpy as np
import sys

from image_functions import (doBrightness, doContrast, doNegative, 
                             doDefault, doVerticalFlip, doHorizontalFlip, 
                             doDiagonalFlip, doShrink, doEnlarge, min_filter, max_filter, adaptive_median_filter)

def print_help():
    help_text = """
    Available Commands:
    
    --help                : Show this help message and exit.
    
    --negative            : Apply a negative filter to the image.
    
    --default             : Reset the image to the original state.
    
    --vflip               : Apply vertical flip to the image.
    
    --hflip               : Apply horizontal flip to the image.
    
    --dflip               : Apply diagonal flip to the image.
    
    --brightness <val>    : Adjust brightness by the specified value. Range: [-255, 255].
                            Example: --brightness 50 increases brightness; --brightness -50 decreases it.
    
    --contrast <val>      : Adjust contrast by the specified factor. Range: [0, 5.0].
                            Example: --contrast 1.2 increases contrast by 20%; --contrast 0.8 decreases it.
    
    --shrink <val>        : Shrink the image by a specific factor. Range: (1.0, 5.0].
                            Example: --shrink 1.5 shrinks the image by 1.5x.
    
    --enlarge <val>       : Enlarge the image by a specific factor. Range: (1.0, 5.0].
                            Example: --enlarge 1.5 enlarges the image by 1.5x.
    
    --min                 : Apply a minimum filter with a kernel to reduce noise.
                            Example: --min (applies a 3x3 minimum filter kernel).
    
    --max                 : Apply a maximum filter with a kernel to enhance details.
                            Example: --max (applies a 3x3 maximum filter kernel).
    
    --adaptive            : Apply adaptive median filtering to reduce noise. Max kernel size: 7.
                            Example: --adaptive (applies adaptive median filtering up to kernel size 7).
    
    --mse                 : Calculate Mean Squared Error between original and noised image.
                            Example: --mse
    
    --pmse                : Calculate Peak Mean Square Error.
                            Example: --pmse
    
    --snr                 : Calculate Signal-to-Noise Ratio.
                            Example: --snr
    
    --psnr                : Calculate Peak Signal-to-Noise Ratio.
                            Example: --psnr
    
    --md                  : Calculate the maximum difference between the original and noised image.
                            Example: --md
    """
    print(help_text)

im = Image.open("./images/lena_8bits.bmp")

im = Image.open("lenac.bmp")
im_noised = Image.open("result.bmp")

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
        case '--min':
            arr = min_filter(arr_noised)
            print("Min filter applied.")
        case '--max':
            arr = max_filter(arr_noised)
            print("Max filter applied.")
        case '--adaptive':
            arr = adaptive_median_filter(arr_noised)
            print("Adaptive median filter applied.")
        case _:
            print("Unknown command: " + command)
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
    else:
        print("Unknown command: " + command)
        sys.exit()


newIm = Image.fromarray(arr.astype(np.uint8))
newIm.save("result.bmp")