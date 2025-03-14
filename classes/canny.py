import cv2
from PyQt5.QtWidgets import QApplication, QFileDialog
import numpy as np
import sys
import random

def apply_canny_edge_detection(img, sigma, low_threshold, high_threshold):
    
    gray_img = convert_rgb_to_gray(img)
    
    # stage 1: noise reduction using the Gaussian filter
    smoothed_image = apply_gaussian_filter(gray_img, filter_size=5, sigma=sigma)
    
    # stage 2: calculate gradient magnitude and orientation
    magnitude, orientation = find_magnitude_and_orientation(smoothed_image)
    
    # stage 3: non-maximum suppression (edge thinning)
    suppressed_image = apply_non_maximum_suppression(magnitude, orientation)
    
    # stage 4: double thresholding to eliminate edges below the low threshold, and identify strong and weak edges
    thresholded_image, edge_ids = apply_double_thresholding(suppressed_image, low_threshold=low_threshold, high_threshold=high_threshold)
    
    # stage 5: edge tracking by hysteresis
    final_edges = apply_hysteresis_thresholding(thresholded_image, edge_ids)

    return final_edges

# def select_image():
#     app = QApplication(sys.argv)
#     options = QFileDialog.Options()
#     file_path, _ = QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.jpg *.jpeg *.png *.bmp *.gif)", options=options)
#     if file_path:
#         img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
#         return img
#     return None

def convert_rgb_to_gray(image):
    '''
    NOTE: RETURNS SINGLE CHANNEL IMAGE
    '''
    if len(image.shape) == 2:
        return image  # already grayscale
    
    r, g, b = cv2.split(image)
    gray_image = 0.299 * r + 0.587 * g + 0.114 * b
    return gray_image.astype(np.uint8)

def convolve(image, kernel):
    '''
    NOTE: THIS CONVOLVE FUNCTION DIFFERS FROM THE ONE IN THE IMAGE CLASS IN THE FOLLOWING:

        - added handling for both single and multi-channel images
        - edited the padding logic for edge cases
    '''

    # convert single channel to 2D if needed
    is_single_channel = len(image.shape) == 2
    if is_single_channel:
        image = image[:, :, np.newaxis]
    
    kernel_size = kernel.shape[0]
    pad_size = kernel_size // 2
    image = image.astype(np.float32)
    channels = image.shape[2]
    
    # pad the image with zeros
    padded_image = np.pad(
        image, 
        pad_width=((pad_size, pad_size), (pad_size, pad_size), (0, 0)), 
        mode='constant', 
        constant_values=0
    )
    
    output_image = np.zeros_like(image, dtype=np.float32)
    
    for c in range(channels):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded_image[i:i+kernel_size, j:j+kernel_size, c]
                output_image[i, j, c] = np.sum(region * kernel)
    
    # return in the same format as input
    if is_single_channel:
        return output_image[:, :, 0]
    return output_image

def apply_gaussian_filter(img, filter_size=3, sigma=1):

    # checking if filter size is odd and if not => make it odd
    if filter_size % 2 == 0:
        filter_size += 1
    
    x, y = np.meshgrid(
        np.arange(-filter_size // 2, (filter_size // 2) + 1),
        np.arange(-filter_size // 2, (filter_size // 2) + 1)
    )
    kernel = np.exp(-(x**2 + y**2)/(2*sigma**2)) / (2*np.pi*sigma**2)
    kernel = kernel / np.sum(kernel)  
    
    output_image = convolve(img, kernel)
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)
    
    return output_image

def find_magnitude_and_orientation(img, filter_size=3):
    
    '''
    Default filter size is 5x5

    Returns 2 matrices: magnitude and orientation
    '''

    if filter_size == 3:
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
    gradient_x = convolve(img, sobel_x)
    gradient_y = convolve(img, sobel_y)
    
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    
    orientation = np.arctan2(gradient_y, gradient_x) * 180 / np.pi # in DEGREES
    
    return magnitude, orientation

def apply_non_maximum_suppression(mag, ang):

    # convert to single channel if needed
    if len(mag.shape) == 3:
        mag = mag[:, :, 0]
    
    if len(ang.shape) == 3:
        ang = ang[:, :, 0]
    
    height, width = mag.shape
    result = np.zeros_like(mag)
    
    for i_y in range(1, height-1):
        for i_x in range(1, width-1):

            grad_ang = ang[i_y, i_x]
            grad_ang = abs(grad_ang-180) if abs(grad_ang) > 180 else abs(grad_ang)
            
            # finding neighboring pixels based on the gradient direction
            if grad_ang <= 22.5 or grad_ang > 157.5:  # horizontal edge
                neighb_1_x, neighb_1_y = i_x-1, i_y
                neighb_2_x, neighb_2_y = i_x+1, i_y
            elif 22.5 < grad_ang <= 67.5:  # diagonal edge (/)
                neighb_1_x, neighb_1_y = i_x-1, i_y-1
                neighb_2_x, neighb_2_y = i_x+1, i_y+1
            elif 67.5 < grad_ang <= 112.5:  # vertical edge
                neighb_1_x, neighb_1_y = i_x, i_y-1
                neighb_2_x, neighb_2_y = i_x, i_y+1
            else:  # diagonal edge (\)
                neighb_1_x, neighb_1_y = i_x-1, i_y+1
                neighb_2_x, neighb_2_y = i_x+1, i_y-1
            
            if (0 <= neighb_1_y < height and 0 <= neighb_1_x < width and
                0 <= neighb_2_y < height and 0 <= neighb_2_x < width):
                
                if (mag[i_y, i_x] >= mag[neighb_1_y, neighb_1_x] and 
                    mag[i_y, i_x] >= mag[neighb_2_y, neighb_2_x]):
                    result[i_y, i_x] = mag[i_y, i_x]
    
    return result

def apply_double_thresholding(mag, low_threshold, high_threshold):

    # convert to single channel if needed
    if len(mag.shape) == 3:
        mag = mag[:, :, 0]
    
    height, width = mag.shape
    
    # 0: non-edge, 1: weak edge, 2: strong edge
    ids = np.zeros_like(mag)
    
    result = np.zeros_like(mag)
    
    for i_y in range(height):
        for i_x in range(width):
            pixel_value = mag[i_y, i_x]
            
            if pixel_value >= high_threshold:
                ids[i_y, i_x] = 2  # strong edge
                result[i_y, i_x] = 255
            elif low_threshold <= pixel_value < high_threshold:
                ids[i_y, i_x] = 1  # weak edge
            # else: non-edge (already 0)
    
    return result, ids

def apply_hysteresis_thresholding(mag, ids):

    # convert to single channel if needed
    if len(mag.shape) == 3:
        mag = mag[:, :, 0]
    
    if len(ids.shape) == 3:
        ids = ids[:, :, 0]
    
    height, width = ids.shape
    result = np.zeros_like(mag)
    
    # set strong edges into an intensity of 255
    result[ids == 2] = 255
    
    dx = [-1, -1, -1, 0, 0, 1, 1, 1]
    dy = [-1, 0, 1, -1, 1, -1, 0, 1]
    
    # recursively find connected edges
    def trace_edge(y, x):
        result[y, x] = 255
        
        for k in range(8):
            ny, nx = y + dy[k], x + dx[k]
            
            if 0 <= ny < height and 0 <= nx < width:
                if ids[ny, nx] == 1 and result[ny, nx] == 0:
                    trace_edge(ny, nx)
    
    # finding all strong edges and tracing connected weak edges
    for i_y in range(height):
        for i_x in range(width):
            if ids[i_y, i_x] == 2 and result[i_y, i_x] == 255:
                trace_edge(i_y, i_x)
    
    return result

def normalize_image(image, new_min=0, new_max=255):
    image = image.astype(np.float32)
    
    old_min = np.min(image)
    old_max = np.max(image)
    
    # avoiding division by zero
    if old_max == old_min:
        return np.ones_like(image) * new_min
    
    scale = (new_max - new_min) / (old_max - old_min)
    normalized = (image - old_min) * scale + new_min
    
    return np.clip(normalized, new_min, new_max).astype(np.uint8)
