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