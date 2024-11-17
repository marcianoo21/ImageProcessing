from PIL import Image
import numpy as np
import sys
import matplotlib.pyplot as plt


image = Image.open('./images/lenac_impulse3.bmp')
# image1 = Image.open('./images/lena.bmp')
smaller_image = image.resize((256, 256))
# arr = np.array(image)
arr = np.array(smaller_image)
width, height, channels = arr.shape
# arr1 = np.array(image1)
# print(width, height, channels)


red_pix_values = []
green_pix_values = []
blue_pix_values = []


for i in range(width):
    for j in range(height):
        red_pix_values.append(arr[i][j][0])
        green_pix_values.append(arr[i][j][1])
        blue_pix_values.append(arr[i][j][2])

# print(len(red_pix_values), len(green_pix_values), len(blue_pix_values))

# for value in red_pix_values:
   
# Red channel histogram
red_hist = plt.hist(red_pix_values, bins=256, range=(0, 256), color='red', alpha=0.7, edgecolor='black')
plt.title('Red Pixel Value Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()
# plt.savefig('red_channel_histogram.png', dpi=300) 
# plt.close()
# x_values = np.arange(256)
# red_freq = red_hist[0]
# int_red_freq = [int(i) for i in red_freq] #comprehension to sie nazywało -> a no tak
# red_pdf = [i/sum(int_red_freq) for i in int_red_freq] 
# print(red_pdf)
# plt.subplot(1, 3, 1)
# plt.plot(x_values, red_pdf, color='red', label='Red PDF')
# plt.title('Red Pixel Value PDF')
# plt.xlabel('Pixel Value')
# plt.ylabel('Probability Density')
# # plt.legend()
# plt.show()

# print(len(red_hist[int(0)]))


# Green channel histogram
green_hist = plt.hist(green_pix_values, bins=256, range=(0, 256), color='green', alpha=0.7, edgecolor='black')
plt.title('Green Pixel Value Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()
plt.savefig('green_channel_histogram.png', dpi=300) 
plt.close()

# print(sum(green_hist[0]))

# Blue channel histogram
plt.hist(blue_pix_values, bins=256, range=(0, 256), color='blue', alpha=0.7, edgecolor='black')
plt.title('Blue Pixel Value Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()
plt.savefig('blue_channel_histogram.png', dpi=300) 
plt.close()


# plt.hist(pix_values)
# plt.imsave('histogram.png')
# plt.show()


# newIm = Image.fromarray(arr.astype(np.uint8))
# newIm.show()

# import numpy as np
# import cv2
# import matplotlib.pyplot as plt

# # Function to calculate and plot histogram
# def calculate_histogram(channel_data):
#     hist, bins = np.histogram(channel_data, bins=256, range=(0, 256))
#     return hist, bins[:-1]

# # Exponential PDF transformation function
# def exponential_transform(pixel_value, lambda_param):
#     return int(255 * (1 - np.exp(-lambda_param * pixel_value / 255)))

# # Apply Exponential Transformation to the image channel
# def apply_exponential_transformation(image, lambda_param=1.0):
#     transformed_image = np.zeros_like(image)
#     for i in range(image.shape[0]):  # Loop through each pixel
#         for j in range(image.shape[1]):
#             transformed_image[i, j] = exponential_transform(image[i, j], lambda_param)
#     return transformed_image

# # Save histogram as an image
# def save_histogram(hist, bins, filename):
#     plt.bar(bins, hist, width=1, color='gray', alpha=0.7, edgecolor='black')
#     plt.title('Histogram')
#     plt.xlabel('Pixel Intensity')
#     plt.ylabel('Frequency')
#     plt.savefig(filename)
#     plt.close()

# # Image Quality Calculation (MSE, PSNR, etc.)
# def calculate_image_quality(original, transformed):
#     mse = np.mean((original - transformed) ** 2)
#     psnr = 10 * np.log10(255**2 / mse)
#     return mse, psnr

# # Main execution
# if __name__ == "__main__":
#     # Load the image
#     image = cv2.imread('./images/lenac.bmp')
    
#     # Extract a channel (for example, the Red channel)
#     red_channel = image[:, :, 2]
    
#     # Calculate original histogram
#     hist, bins = calculate_histogram(red_channel)
    
#     # Save original histogram
#     save_histogram(hist, bins, 'original_histogram.png')
    
#     # Apply Exponential Transformation to the Red channel
#     lambda_param = 1.0  # Adjust this parameter for stronger/weaker enhancement
#     transformed_red_channel = apply_exponential_transformation(red_channel, lambda_param)
    
#     # Replace the Red channel with the transformed channel
#     image[:, :, 2] = transformed_red_channel
    
#     # Calculate transformed histogram
#     hist_transformed, _ = calculate_histogram(transformed_red_channel)
    
#     # Save transformed histogram
#     save_histogram(hist_transformed, bins, 'transformed_histogram.png')
    
#     # Calculate image quality metrics
#     mse, psnr = calculate_image_quality(red_channel, transformed_red_channel)
    
#     print(f'Mean Squared Error (MSE): {mse}')
#     print(f'Peak Signal-to-Noise Ratio (PSNR): {psnr}')
    
#     # Save the enhanced image
#     cv2.imwrite('enhanced_image.jpg', image)
