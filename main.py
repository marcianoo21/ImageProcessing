from PIL import Image
import numpy as np
import sys

from image_functions import (doBrightness, doContrast, doNegative, 
                             doDefault, doVerticalFlip, doHorizontalFlip, 
                             doDiagonalFlip, doShrink, doEnlarge, min_filter, max_filter, adaptive_median_filter, mse, pmse, snr, psnr, max_diff,
                             mean, variance, std_dev, variation_coeff_1, asymmetry_coeff, flattening_coeff, variation_coeff_2, entropy, optimized_laplacian_filter, universal_laplacian_filter, roberts_operator_ii, create_histogram, exponential_density_function)

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
    
    --slaplace            : Apply Laplacian filter to the image.
                            Example: --slaplace

    --olaplace             : Apply Laplacian optimized filter to the image.
                             Example: --olaplace
    
    --cmean               : Compute and display the mean of the image.
                            Example: --cmean
    
    --cvariance           : Compute and display the variance of the image.
                            Example: --cvariance
    
    --cstdev              : Compute and display the standard deviation of the image.
                            Example: --cstdev
    
    --cvarcoi             : Compute and display the variation coefficient I of the image.
                            Example: --cvarcoi
    
    --casyco              : Compute and display the asymmetry coefficient of the image.
                            Example: --casyco
    
    --cflattening         : Compute and display the flattening coefficient of the image.
                            Example: --cflattening
    
    --cvarcoii            : Compute and display the variation coefficient II of the image.
                            Example: --cvarcoii
    
    --centropy            : Compute and display the entropy of the image.
                            Example: --centropy
    """
    print(help_text)

def apply_command(command, param, arr, arr_noised):
    kernel = np.array(
    [[0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0]])
    if command == '--help':
        print_help()
        sys.exit()
    elif command == '--negative':
        return doNegative(arr)
    elif command == '--default':
        return doDefault(arr)
    elif command == '--vflip':
        return doVerticalFlip(arr)
    elif command == '--hflip':
        return doHorizontalFlip(arr)
    elif command == '--dflip':
        return doDiagonalFlip(arr)
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
    elif command == '--brightness':
        return doBrightness(arr, int(param))
    elif command == '--contrast':
        return doContrast(arr, float(param))
    elif command == '--shrink':
        return doShrink(arr, int(param))
    elif command == '--enlarge':
        return doEnlarge(arr, int(param))
    elif command == '--adaptive':
        return adaptive_median_filter(arr, int(param))
    elif command == '--min':
        return min_filter(arr, int(param))
    elif command == '--max':
        return max_filter(arr, int(param))
    elif command == '--slaplace':
        return universal_laplacian_filter(arr, kernel)
    elif command == '--olaplace':
        return optimized_laplacian_filter(arr)
    elif command == '--orobertsii':
        return roberts_operator_ii(arr)
    elif command == '--histogram':
        return create_histogram(arr)
    elif command == '--hexponent':
        return exponential_density_function(arr, int(sys.argv[3]) , int(sys.argv[4]) )
    elif command in ['--cmean', '--cvariance', '--cstdev', '--cvarcoi', '--casyco', '--cflattening', '--cvarcoii', '--centropy']:
        histogram, _ = np.histogram(arr, bins=256, range=(0, 256))
        total_pixels = arr.size
        if command == '--cmean':
            mean_value = mean(histogram, total_pixels)
            print("Mean:", mean_value)
        elif command == '--cvariance':
            mean_value = mean(histogram, total_pixels)
            variance_value = variance(histogram, total_pixels, mean_value)
            print("Variance:", variance_value)
        elif command == '--cstdev':
            mean_value = mean(histogram, total_pixels)
            variance_value = variance(histogram, total_pixels, mean_value)
            std_dev_value = std_dev(variance_value)
            print("Standard Deviation:", std_dev_value)
        elif command == '--cvarcoi':
            mean_value = mean(histogram, total_pixels)
            variance_value = variance(histogram, total_pixels, mean_value)
            std_dev_value = std_dev(variance_value)
            variation_coeff_1_value = variation_coeff_1(std_dev_value, mean_value)
            print("Variation Coefficient I:", variation_coeff_1_value)
        elif command == '--casyco':
            mean_value = mean(histogram, total_pixels)
            variance_value = variance(histogram, total_pixels, mean_value)
            std_dev_value = std_dev(variance_value)
            asymmetry_coeff_value = asymmetry_coeff(histogram, total_pixels, mean_value, std_dev_value)
            print("Asymmetry Coefficient:", asymmetry_coeff_value)
        elif command == '--cflattening':
            mean_value = mean(histogram, total_pixels)
            variance_value = variance(histogram, total_pixels, mean_value)
            std_dev_value = std_dev(variance_value)
            flattening_coeff_value = flattening_coeff(histogram, total_pixels, mean_value, std_dev_value)
            print("Flattening Coefficient:", flattening_coeff_value)
        elif command == '--cvarcoii':
            variation_coeff_2_value = variation_coeff_2(histogram, total_pixels)
            print("Variation Coefficient II:", variation_coeff_2_value)
        elif command == '--centropy':
            entropy_value = entropy(histogram, total_pixels)
            print("Entropy:", entropy_value)
    else:
        print("Unknown command: " + command)
        sys.exit()
    return arr


if len(sys.argv) < 2:
    print("No command line parameters given.")
    print_help()
    sys.exit()

image_path = sys.argv[1]

if len(sys.argv) < 3:
    print("No command specified.")
    print_help()
    sys.exit()

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
else:
    param = None
    arr_noised = None

arr = apply_command(command, param, arr1, arr_noised)

if command not in ['--mse', '--pmse', '--snr', '--psnr', '--md', '--cmean', '--cvariance', '--cstdev', '--cvarcoi', '--casyco', '--cflattening', '--cvarcoii', '--centropy']:
    newIm = Image.fromarray(arr.astype(np.uint8))
    newIm.save("result.bmp")
    print("Output saved as 'result.bmp'")