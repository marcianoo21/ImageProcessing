from PIL import Image
import numpy as np
import sys

from image_functions import (doBrightness, doContrast, doNegative, 
                             doDefault, doVerticalFlip, doHorizontalFlip, 
                             doDiagonalFlip, doShrink, doEnlarge, min_filter, max_filter, adaptive_median_filter, mse, pmse, snr, psnr, max_diff)

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


if len(sys.argv) < 2:
    print("No command line parameters given.")
    print_help()
    sys.exit()

image_path = sys.argv[1]
command = sys.argv[2]

# Open the main image
try:
    im = Image.open(image_path)
except Exception as e:
    print(f"Failed to open image: {image_path}. Error: {e}")
    sys.exit()

arr1 = np.array(im)

# Check if additional parameter (value or second image) is provided
if len(sys.argv) == 4:
    param = sys.argv[3]
    if command in ['--mse', '--pmse', '--snr', '--psnr', '--md']:
        try:
            im_noised = Image.open(param)
            im_noised_resized = im_noised.resize(im.size)
            arr_noised = np.array(im_noised_resized)
        except Exception as e:
            print(f"Failed to open comparison image: {param}. Error: {e}")
            sys.exit()
    else:
        # Convert param to numerical type for commands that need it
        try:
            param = float(param) if '.' in param else int(param)
        except ValueError:
            print(f"Invalid parameter value: {param}. Expected a number.")
            sys.exit()

# Image processing based on command
if command == '--help':
    print_help()
    sys.exit()
elif command == '--negative':
    arr = doNegative(arr1)
elif command == '--default':
    arr = doDefault(arr1)
elif command == '--vflip':
    arr = doVerticalFlip(arr1)
elif command == '--hflip':
    arr = doHorizontalFlip(arr1)
elif command == '--dflip':
    arr = doDiagonalFlip(arr1)
elif command == '--min':
    arr = min_filter(arr1)
    print("Min filter applied.")
elif command == '--max':
    arr = max_filter(arr1)
    print("Max filter applied.")
elif command == '--adaptive':
    arr = adaptive_median_filter(arr1)
    print("Adaptive median filter applied.")
elif command == '--brightness':
    arr = doBrightness(param, arr1)
elif command == '--contrast':
    arr = doContrast(param, arr1)
elif command == '--shrink':
    arr = doShrink(param, arr1)
elif command == '--enlarge':
    arr = doEnlarge(param, arr1)
elif command == '--mse':
    mse_value = mse(arr1, arr_noised)
    print("Mean Squared Error: " + str(mse_value))
elif command == '--pmse':
    pmse_value = pmse(arr1, arr_noised)
    print("Peak mean square error: " + str(pmse_value))
elif command == '--snr':
    snr_value = snr(arr1, arr_noised)
    print("Signal to noise ratio: " + str(snr_value))
elif command == '--psnr':
    psnr_value = psnr(arr1, arr_noised)
    print("Peak signal to noise ratio: " + str(psnr_value))
elif command == '--md':
    md_value = max_diff(arr1, arr_noised)
    print("Max difference: " + str(md_value))
else:
    print("Unknown command: " + command)
    sys.exit()

# Save the resulting image only if an image transformation was applied
if command not in ['--mse', '--pmse', '--snr', '--psnr', '--md']:
    newIm = Image.fromarray(arr.astype(np.uint8))
    newIm.save("result.bmp")
    print("Output saved as 'result.bmp'")