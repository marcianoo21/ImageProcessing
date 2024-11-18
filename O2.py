import cv2
import numpy as np
import argparse
import os

def roberts_operator_ii(image):

    height, width = image.shape
    
    filtered_image = np.zeros_like(image, dtype=np.float32)
    
    for n in range(height - 1):  # Exclude the last row
        for m in range(width - 1):  # Exclude the last column
            diff1 = abs(float(image[n, m]) - float(image[n + 1, m + 1]))
            diff2 = abs(float(image[n, m + 1]) - float(image[n + 1, m]))
            
            filtered_image[n, m] = diff1 + diff2
    
    filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX)
    return filtered_image.astype(np.uint8)

if __name__ == "__main__":
  
    parser = argparse.ArgumentParser(description="Apply Roberts Operator II for edge detection.")
    parser.add_argument("--input", required=True, help="Path to the input grayscale image.")
    parser.add_argument("--output", required=True, help="Path to save the filtered image.")
    args = parser.parse_args()
    
    image = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image: {args.input}")
    
    filtered_image = roberts_operator_ii(image)
    
    cv2.imwrite(args.output, filtered_image)
    print(f"Filtered image saved at {args.output}")
