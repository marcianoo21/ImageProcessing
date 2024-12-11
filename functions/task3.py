import numpy as np
from collections import deque



def erosion(image, struct_elem):

    k_height, k_width = struct_elem.shape
    pad_height, pad_width = k_height // 2, k_width // 2

    # Dodaj obramowanie do obrazu
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
    result = np.zeros_like(image)

    # Przesuwamy element strukturalny po obrazie
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Wycinamy fragment obrazu o rozmiarze elementu strukturalnego
            region = padded_image[i:i + k_height, j:j + k_width]
            # Sprawdzamy czy cały element strukturalny mieści się w regionie
            if np.all(region & struct_elem == struct_elem):
                result[i, j] = 1

    return result

def dilation(image, struct_elem):

    k_height, k_width = struct_elem.shape
    pad_height, pad_width = k_height // 2, k_width // 2

    # Dodaj obramowanie do obrazu
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
    result = np.zeros_like(image)

    # Przesuwamy element strukturalny po obrazie
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Wycinamy fragment obrazu o rozmiarze elementu strukturalnego
            region = padded_image[i:i + k_height, j:j + k_width]
            # Sprawdzamy czy zachodzi jakakolwiek część wspólna
            if np.any(region & struct_elem):
                result[i, j] = 1

    return result

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

def bitwise_and(image1, image2):
    return np.logical_and(image1, image2)

def bitwise_not(image):
    return np.logical_not(image)

def subtract(image1, image2):
    return np.logical_and(image1, np.logical_not(image2))

def hmt(image, struct_elem1, struct_elem2):
    erosion_image1 = erosion(image, struct_elem1)
    complement_image = bitwise_not(image)
    erosion_image2 = erosion(complement_image, struct_elem2)
    return bitwise_and(erosion_image1, erosion_image2)

# (A ⊕ B) \ A -dilation subtracts A
def operation_1(image, struct_elem):
    dilated_image = dilation(image, struct_elem)
    return subtract(dilated_image, image)

# A \ (A ⊖ B) -A subtracts erosion
def operation_2(image, struct_elem):
    eroded_image = erosion(image, struct_elem)
    return subtract(image, eroded_image)

# (A ⊕ B) \ (A ⊖ B) -dilation subtracts erosion
def operation_3(image, struct_elem):
    dilated_image = dilation(image, struct_elem)
    eroded_image = erosion(image, struct_elem)
    return subtract(dilated_image, eroded_image)

def region_growing_rgb(image, seed, threshold):
  
    rows, cols, _ = image.shape
    segmented = np.zeros((rows, cols), dtype=bool)
    queue = deque([seed])
    seed_color = image[seed]  # The RGB color at the seed point
    
    while queue:
        x, y = queue.popleft()
        
        if segmented[x, y]:
            continue
        
        segmented[x, y] = True
        
        # Check 8-neighbor pixels
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and not segmented[nx, ny]:
                neighbor_color = image[nx, ny]
                # Calculate Euclidean distance in RGB space
                color_diff = np.linalg.norm(neighbor_color - seed_color)
                if color_diff <= threshold:
                    queue.append((nx, ny))
    
    return segmented.astype(np.uint8) * 255  # Convert boolean mask to binary image