import cv2
import numpy as np

# Process of expending images, increasing brightness of the image if original image and pixel matches we replace part of image with mask
def dilation(image, struct_elem):
    return cv2.dilate(image, struct_elem)
    # A∩B  - część wspólna dwóch setów

# Opposite process of dilation, shrinking image, if original image and pixel matches we replace only this pixels, the rest is 0
def erosion(image, struct_elem):
    return cv2.erode(image, struct_elem)
    # A⊆B - A jest subsetem B (wszystkie elementy z A muszą zawierać się w B)

# erosion is followed by dilation operation
# to identify gaps in image
# to make edges sharper or smoother
# isolating touching images
def opening(image, struct_elem):
    erosion_image = erosion(image, struct_elem)
    return dilation(erosion_image, struct_elem)
    

# dilation is followed by erosion
# to eliminate small holes
def closing(image, struct_elem):
    dilation_image = dilation(image, struct_elem)
    return erosion(dilation_image, struct_elem)


def hmt(image, struct_elem1, struct_elem2):
    erosion_image1 = erosion(image, struct_elem1)
    complement_image = cv2.bitwise_not(image)
    erosion_image2 = erosion(complement_image, struct_elem2)
    return cv2.bitwise_and(erosion_image1, erosion_image2)

# (A ⊕ B) \ A -dilation subtracts A
def operation_1(image, struct_elem):
    dilated_image = dilation(image, struct_elem)
    return cv2.subtract(dilated_image, image)

# A \ (A ⊖ B) -A subtracts erosion
def operation_2(image, struct_elem):
    eroded_image = erosion(image, struct_elem)
    return cv2.subtract(image, eroded_image)

# (A ⊕ B) \ (A ⊖ B) -dilation subtracts erosion
def operation_3(image, struct_elem):
    dilated_image = dilation(image, struct_elem)
    eroded_image = erosion(image, struct_elem)
    return cv2.subtract(dilated_image, eroded_image)