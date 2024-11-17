import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# Define the main functionality
class ImageProcessor:
    def __init__(self, image_path, output_dir):
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.output_dir = output_dir
        self.histogram = None
        self.normalized_histogram = None
        self.cdf = None
        self.image_characteristics = {}

        if self.image is None:
            raise ValueError(f"Could not read the image: {image_path}")

        os.makedirs(output_dir, exist_ok=True)

    def calculate_histogram(self):
        hist = cv2.calcHist([self.image], [0], None, [256], [0, 256])
        self.histogram = hist.flatten()
        self.normalized_histogram = self.histogram / self.histogram.sum()

        # Save histogram as an image
        plt.figure()
        plt.plot(self.histogram)
        plt.title("Histogram")
        plt.xlabel("Intensity Value")
        plt.ylabel("Frequency")
        histogram_path = os.path.join(self.output_dir, "histogram.png")
        plt.savefig(histogram_path)
        plt.close()
        print(f"Histogram saved at {histogram_path}")

    def calculate_cdf(self):
        if self.normalized_histogram is None:
            raise ValueError("Normalized histogram not calculated yet.")

        self.cdf = np.cumsum(self.normalized_histogram)

    def apply_quality_improvement(self, method):
        if self.cdf is None:
            self.calculate_cdf()

        # Choose the transformation function based on the method
        if method == "huniform":
            g = np.arange(256)  # Uniform mapping
        elif method == "hexponent":
            g = 255 * (1 - np.exp(-2 * self.cdf))
        elif method == "hrayleigh":
            g = 255 * np.sqrt(-np.log(1 - self.cdf))
        elif method == "hpower":
            g = 255 * np.power(self.cdf, 2 / 3)
        elif method == "hhyper":
            g = 255 * (self.cdf / (1 - self.cdf + 1e-8))
        else:
            raise ValueError(f"Unknown method: {method}")

        g = np.clip(g, 0, 255).astype(np.uint8)

        # Map the image using the calculated transformation
        transformed_image = g[self.image]

        # Save the transformed image
        output_path = os.path.join(self.output_dir, f"transformed_{method}.png")
        cv2.imwrite(output_path, transformed_image)
        print(f"Transformed image saved at {output_path}")

        return transformed_image

    def calculate_image_characteristics(self):
        if self.histogram is None:
            self.calculate_histogram()

        H = self.normalized_histogram
        N = H.sum()
        m = np.arange(256)

        # Mean
        mean = np.sum(m * H) / N
        self.image_characteristics['mean'] = mean

        # Variance
        variance = np.sum((m - mean) ** 2 * H) / N
        self.image_characteristics['variance'] = variance

        # Standard Deviation
        stdev = np.sqrt(variance)
        self.image_characteristics['stdev'] = stdev

        # Variation Coefficient I
        varcoi = stdev / mean if mean != 0 else 0
        self.image_characteristics['varcoi'] = varcoi

        # Asymmetry Coefficient
        asymmetry = np.sum((m - mean) ** 3 * H) / (N * stdev ** 3)
        self.image_characteristics['asymmetry'] = asymmetry

        # Flattening Coefficient
        flattening = np.sum((m - mean) ** 4 * H) / (N * stdev ** 4) - 3
        self.image_characteristics['flattening'] = flattening

        # Variation Coefficient II
        varcoii = (1 / N) ** 2 * np.sum(H ** 2)
        self.image_characteristics['varcoii'] = varcoii

        # Information Source Entropy
        entropy = -np.sum(H * np.log2(H + 1e-8))
        self.image_characteristics['entropy'] = entropy

    def display_image_characteristics(self):
        if not self.image_characteristics:
            self.calculate_image_characteristics()

        print("Image Characteristics:")
        for key, value in self.image_characteristics.items():
            print(f"  {key}: {value}")

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Image Quality Improvement based on Histogram.")
    parser.add_argument("--input", required=True, help="Path to the input image.")
    parser.add_argument("--output", required=True, help="Directory to save outputs.")
    parser.add_argument("--method", required=True, choices=["huniform", "hexponent", "hrayleigh", "hpower", "hhyper"],
                        help="Histogram quality improvement method.")
    parser.add_argument("--histogram", action="store_true", help="Save histogram.")
    parser.add_argument("--characteristics", action="store_true", help="Calculate image characteristics.")
    return parser.parse_args()

# Main entry point
if __name__ == "__main__":
    args = parse_args()

    processor = ImageProcessor(args.input, args.output)

    if args.histogram:
        processor.calculate_histogram()

    if args.characteristics:
        processor.calculate_image_characteristics()
        processor.display_image_characteristics()

    # Apply the chosen method
    processor.apply_quality_improvement(args.method)
