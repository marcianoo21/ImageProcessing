import cv2
import numpy as np
import argparse
import os

def universal_convolution(image, mask):
    k = mask.shape[0] // 2  # Assuming square mask
    
    padded_image = cv2.copyMakeBorder(image, k, k, k, k, cv2.BORDER_REPLICATE)
    
    filtered_image = np.zeros_like(image, dtype=np.float32)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            roi = padded_image[i:i + mask.shape[0], j:j + mask.shape[1]]
            filtered_image[i, j] = np.sum(roi * mask)
    
    return cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def optimized_laplacian_filter(image):
    mask = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ], dtype=np.float32)
    
    filtered_image = cv2.filter2D(image, -1, mask, borderType=cv2.BORDER_REPLICATE)
    
    return filtered_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear Image Filtration in Spatial Domain.")
    parser.add_argument("--input", required=True, help="Path to the input grayscale image.")
    parser.add_argument("--output", required=True, help="Directory to save output images.")
    parser.add_argument("--mask", type=int, choices=[1, 2, 3], required=True, 
                        help="Laplacian mask choice: 1, 2, or 3.")
    args = parser.parse_args()
    
    image = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image: {args.input}")
    
    masks = {
        1: np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32),
        2: np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32),
        3: np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32),
    }
    
    os.makedirs(args.output, exist_ok=True)
    
    universal_filtered = universal_convolution(image, masks[args.mask])
    universal_output_path = os.path.join(args.output, f"universal_filtered_mask{args.mask}.png")
    cv2.imwrite(universal_output_path, universal_filtered)
    print(f"Universal filtered image saved at {universal_output_path}")
    
    optimized_filtered = optimized_laplacian_filter(image)
    optimized_output_path = os.path.join(args.output, f"optimized_filtered_mask{args.mask}.png")
    cv2.imwrite(optimized_output_path, optimized_filtered)
    print(f"Optimized filtered image saved at {optimized_output_path}")
