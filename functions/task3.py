import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from PIL import Image

# Process of expending images, increasing brightness of the image if original image and pixel matches we replace part of image with mask
def dilation(image, struct_elem):
    """Perform dilation on a binary image using the given structural element."""
    return binary_dilation(image, structure=struct_elem).astype(int)

# Opposite process of dilation, shrinking image, if original image and pixel matches we replace only this pixels, the rest is 0
def erosion(image, struct_elem):
    """Perform erosion on a binary image using the given structural element."""
    return binary_erosion(image, structure=struct_elem).astype(int)

# erosion is followed by dilation operation
# to identify gaps in image
# to make edges sharper or smoother
# isolating touching images
def opening(image, struct_elem):
    """Perform opening (erosion followed by dilation) on a binary image."""
    eroded = erosion(image, struct_elem)
    return dilation(eroded, struct_elem)

# dilation is followed by erosion
# to eliminate small holes
def closing(image, struct_elem):
    """Perform closing (dilation followed by erosion) on a binary image."""
    dilated = dilation(image, struct_elem)
    return erosion(dilated, struct_elem)

def set_difference(image1, image2):
    """Perform set difference (image1 - image2) between two binary images."""
    return np.logical_and(image1, np.logical_not(image2)).astype(int)

# Create a sample binary image
im = Image.open("images/lena.bmp")
image = np.array(im)
# Define the structural element (iii)
struct_elem = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
], dtype=int)

# Perform morphological operations
dilated_image = dilation(image, struct_elem)
eroded_image = erosion(image, struct_elem)
opened_image = opening(image, struct_elem)
closed_image = closing(image, struct_elem)

# Perform specified operations
operation1 = set_difference(dilated_image, image)  # (A⊕B)∖A
operation2 = set_difference(image, eroded_image)  # A∖(A⊖B)
operation3 = set_difference(dilated_image, eroded_image)  # (A⊕B)∖(A⊖B)

# Print the results
print("Original Image:\n", image)
print("Dilated Image:\n", dilated_image)
print("Eroded Image:\n", eroded_image)
print("Opened Image:\n", opened_image)
print("Closed Image:\n", closed_image)
print("(A⊕B)∖A:\n", operation1)
print("A∖(A⊖B):\n", operation2)
print("(A⊕B)∖(A⊖B):\n", operation3)


# SAMPLE VERSION