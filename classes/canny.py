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


# New Functions for Detecting Geometric Shapes

def detect_shapes(image, edges, detect_lines=True, detect_circles=True, detect_ellipses=True, 
                 line_vote_threshold_percent=50, circle_vote_threshold_percent=50, ellipse_vote_threshold_percent=50):
    """
    Detect lines, circles, and ellipses in an edge-detected image using Hough transforms.
        
    Returns:
    dict: Dictionary containing detected 'lines', 'circles', and 'ellipses'
    """
    results = {}
    
    # Ensure edges is binary and single channel
    if len(edges.shape) == 3:
        edges = edges[:, :, 0]
    
    # Threshold the edge image to make sure it's binary
    _, binary_edges = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)
    
    if detect_lines:
        results['lines'] = detect_lines_hough(binary_edges, line_vote_threshold_percent)
    
    if detect_circles:
        results['circles'] = detect_circles_hough(binary_edges, circle_vote_threshold_percent)
    
    if detect_ellipses:
        results['ellipses'] = detect_ellipses_hough(image)
    
    return results

def detect_lines_hough(edges, vote_threshold_percent=50):
    """
    Detect lines in an edge image using Hough transform.
    
    Returns:
    list: List of detected lines in (rho, theta, votes) format
    """
    height, width = edges.shape
    diagonal = int(np.sqrt(height**2 + width**2))
    
    # Rho range: -diagonal to diagonal with step 1
    rho_range = np.arange(-diagonal, diagonal + 1, 1)
    # Theta range: -90 to 90 degrees with step 1 degree
    theta_range = np.arange(-90, 90, 1) * np.pi / 180
    
    # Initialize the accumulator
    accumulator = np.zeros((len(rho_range), len(theta_range)), dtype=np.int32)
    
    # Mapping of (rho, theta) indices to (x, y) coordinates
    edge_points = np.argwhere(edges > 0)
    
    # Voting process
    for y, x in edge_points:
        for theta_idx, theta in enumerate(theta_range):
            # Calculate rho = x*cos(theta) + y*sin(theta)
            rho = int(x * np.cos(theta) + y * np.sin(theta))
            # Map rho to its index in the accumulator
            rho_idx = rho + diagonal
            if 0 <= rho_idx < len(rho_range):
                accumulator[rho_idx, theta_idx] += 1
    
    # Finding local maxima in the accumulator
    detected_lines = []
    
    # Calculate vote threshold based on percentage of maximum votes
    max_votes = np.max(accumulator)
    if max_votes == 0:
        return []  # No lines detected
    
    vote_threshold = int(max_votes * vote_threshold_percent / 100)
    
    # Non-maxima suppression and threshold application
    for rho_idx in range(1, len(rho_range) - 1):
        for theta_idx in range(1, len(theta_range) - 1):
            votes = accumulator[rho_idx, theta_idx]
            
            # Check if it's above threshold
            if votes > vote_threshold:
                is_local_max = True
                
                # Check 8-neighborhood for non-maxima suppression
                for dr in [-1, 0, 1]:
                    for dt in [-1, 0, 1]:
                        if dr == 0 and dt == 0:
                            continue
                        if accumulator[rho_idx + dr, theta_idx + dt] > votes:
                            is_local_max = False
                            break
                    if not is_local_max:
                        break
                
                if is_local_max:
                    rho = rho_range[rho_idx]
                    theta = theta_range[theta_idx]
                    detected_lines.append((rho, theta, votes))
    
    # Sort lines by vote count (descending)
    detected_lines.sort(key=lambda x: x[2], reverse=True)
    final_lines = merge_similar_lines(detected_lines)
    
    return final_lines


def merge_similar_lines(lines, rho_threshold=10, theta_threshold=np.pi/36):  # theta_threshold ≈ 5 degrees
    if not lines:
        return []
    
    merged_lines = []
    lines = sorted(lines, key=lambda x: x[2], reverse=True)  # Sort by votes
    
    used = [False] * len(lines)
    
    for i, (rho1, theta1, votes1) in enumerate(lines):
        if used[i]:
            continue
            
        used[i] = True
        similar_lines = [(rho1, theta1, votes1)]
        
        for j, (rho2, theta2, votes2) in enumerate(lines[i+1:], i+1):
            if used[j]:
                continue
                
            # Check if lines are similar
            if (abs(rho1 - rho2) < rho_threshold and 
                (abs(theta1 - theta2) < theta_threshold or 
                 abs(abs(theta1 - theta2) - np.pi) < theta_threshold)):
                used[j] = True
                similar_lines.append((rho2, theta2, votes2))
        
        # Average the parameters of similar lines, weighted by votes
        total_votes = sum(line[2] for line in similar_lines)
        avg_rho = sum(line[0] * line[2] for line in similar_lines) / total_votes
        
        # Careful with theta averaging - need to handle wraparound
        sin_avg = sum(np.sin(line[1]) * line[2] for line in similar_lines) / total_votes
        cos_avg = sum(np.cos(line[1]) * line[2] for line in similar_lines) / total_votes
        avg_theta = np.arctan2(sin_avg, cos_avg)
        
        merged_lines.append((avg_rho, avg_theta, total_votes))
    
    return merged_lines


def detect_circles_hough(edges, vote_threshold_percent=50):
    """
    Detect circles in an edge image using Hough transform.
    
    Returns:
    list: List of detected circles in (center_x, center_y, radius, votes) format
    """
    height, width = edges.shape
    
    # Define the range of possible radii
    min_radius = 10
    max_radius = 100
    radius_step = 5
    radius_range = np.arange(min_radius, max_radius + 1, radius_step)  
    
    # Initialize the accumulator (3D: y, x, radius)
    accumulator = np.zeros((height, width, len(radius_range)), dtype=np.int32)
    
    # Get edge points
    edge_points = np.argwhere(edges > 0)
    degree_step = 10  
    # For each edge point, vote in the accumulator
    for y, x in edge_points:
        # For each possible radius
        for r_idx, radius in enumerate(radius_range):
            # For each angle (full circle: 0 to 360 degrees)
            for angle in range(0, 360, degree_step):  
                # Convert angle to radians
                theta = angle * np.pi / 180
                
                # Calculate possible center coordinates
                center_x = int(x - radius * np.cos(theta))
                center_y = int(y - radius * np.sin(theta))
                
                # Check if the center is within image boundaries
                if 0 <= center_y < height and 0 <= center_x < width:
                    accumulator[center_y, center_x, r_idx] += 1
    
    # Finding local maxima in the accumulator
    detected_circles = []
    
    # Calculate vote threshold based on percentage of maximum votes
    max_votes = np.max(accumulator)
    if max_votes == 0:
        return []  # No circles detected
    
    vote_threshold = int(max_votes * vote_threshold_percent / 100)
    
    # Find circles above the threshold
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            for r_idx in range(len(radius_range)):
                votes = accumulator[y, x, r_idx]
                
                # Check if it's above threshold
                if votes > vote_threshold:
                    is_local_max = True
                    
                    # Check neighborhood for non-maxima suppression 
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            for dr in [-1, 0, 1]:
                                if dy == 0 and dx == 0 and dr == 0:
                                    continue
                                ny, nx, nr = y + dy, x + dx, r_idx + dr
                                if (0 <= ny < height and 0 <= nx < width and 
                                    0 <= nr < len(radius_range) and 
                                    accumulator[ny, nx, nr] > votes):
                                    is_local_max = False
                                    break
                            if not is_local_max:
                                break
                        if not is_local_max:
                            break
                    
                    if is_local_max:
                        radius = radius_range[r_idx]
                        detected_circles.append((x, y, radius, votes))
    
    # Sort circles by vote count (descending)
    detected_circles.sort(key=lambda x: x[3], reverse=True)
    
    return detected_circles


# def detect_ellipses_hough(edges, vote_threshold_percent=50):
#     """
#     Detect ellipses in an edge image using Hough transform.
    
#     Returns:
#     list: List of detected ellipses in (center_x, center_y, a, b, angle, votes) format
#     """
#     height, width = edges.shape
    
#     # Define parameter ranges
#     a_range = np.arange(10, 60, 5)  # Major axis
#     b_range = np.arange(5, 50, 5)   # Minor axis
#     angle_range = np.arange(0, 180, 10) * np.pi / 180    # Orientation in radians
    
#     # Initialize the accumulator (5D: center_y, center_x, a, b, angle)
#     # This is a very big accumulator, so we use a dictionary for sparse storage
#     accumulator = {}
    
#     # Get edge points and their gradients
#     edge_points = np.argwhere(edges > 0)
    
#     # For gradient estimation, use Sobel operators
#     grad_x = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=3)
#     grad_y = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=3)
    
#     # Voting process - simplified for computational efficiency
#     # For each edge point and each set of ellipse parameters
#     for i, (y, x) in enumerate(edge_points):
#         # Calculate gradient direction at this point
#         gx = grad_x[y, x]
#         gy = grad_y[y, x]
#         gradient_mag = np.sqrt(gx*gx + gy*gy)
        
#         if gradient_mag > 0:
#             # Normalized gradient direction
#             nx = gx / gradient_mag
#             ny = gy / gradient_mag
            
#             # For each major axis length
#             for a in a_range:
#                 # For computational efficiency, only check a subset of b values
#                 for b_idx, b in enumerate(b_range):
#                     if b > a:  # Major axis should be >= minor axis
#                         continue
                    
#                     # For each orientation
#                     for angle in angle_range:
#                         # Ellipse parametric equations
#                         # Calculate possible center coordinates based on gradient direction
#                         cos_angle = np.cos(angle)
#                         sin_angle = np.sin(angle)
                        
#                         # We use gradient information to constrain center votes
#                         # Based on the fact that the gradient points towards the center
#                         for t in [-1, 1]:  # Try both directions of the gradient
#                             # Calculate parameters for ellipse equation
#                             x_dir = t * nx
#                             y_dir = t * ny
                            
#                             # Transform to ellipse coordinate system
#                             x_rot = x_dir * cos_angle + y_dir * sin_angle
#                             y_rot = -x_dir * sin_angle + y_dir * cos_angle
                            
#                             # Scale to account for ellipse shape
#                             x_rot *= a
#                             y_rot *= b
                            
#                             # Transform back to image coordinates
#                             dx = x_rot * cos_angle - y_rot * sin_angle
#                             dy = x_rot * sin_angle + y_rot * cos_angle
                            
#                             # Calculate center
#                             center_x = int(x + dx)
#                             center_y = int(y + dy)
                            
#                             # Check if center is within image boundaries
#                             if 0 <= center_y < height and 0 <= center_x < width:
#                                 # Convert parameters to indices
#                                 a_idx = np.argmin(np.abs(a_range - a))
#                                 angle_idx = np.argmin(np.abs(angle_range - angle))
                                
#                                 # Vote in accumulator
#                                 key = (center_y, center_x, a_idx, b_idx, angle_idx)
#                                 if key in accumulator:
#                                     accumulator[key] += 1
#                                 else:
#                                     accumulator[key] = 1
    
#     # Find the maximum number of votes
#     max_votes = 0
#     if accumulator:
#         max_votes = max(accumulator.values())
    
#     if max_votes == 0:
#         return []  # No ellipses detected
    
#     # Calculate vote threshold
#     vote_threshold = int(max_votes * vote_threshold_percent / 100)
    
#     # Find ellipses above the threshold
#     detected_ellipses = []
#     for key, votes in accumulator.items():
#         if votes > vote_threshold:
#             center_y, center_x, a_idx, b_idx, angle_idx = key
#             a = a_range[a_idx]
#             b = b_range[b_idx]
#             angle = angle_range[angle_idx] * 180 / np.pi  # Convert back to degrees
#             detected_ellipses.append((center_x, center_y, a, b, angle, votes))
    
#     # Sort ellipses by vote count (descending)
#     detected_ellipses.sort(key=lambda x: x[5], reverse=True)
    
#     return detected_ellipses



def detect_ellipses_hough(image):
    """
    Detect ellipses in an image using Hough transform implementation.
        
    Returns:
        list: List of detected ellipses in ((center_x, center_y), (major_axis, minor_axis), angle) format
    """
     # Convert to grayscale if necessary
    if len(image.shape) == 3:
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Create a binary mask where colored regions are white
        _, saturation, _ = cv2.split(hsv)
        # Threshold on saturation to identify colored regions
        _, mask = cv2.threshold(saturation, 20, 255, cv2.THRESH_BINARY)
    else:
        # If grayscale, use intensity thresholding
        _, mask = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter small contours
    min_area = 10
    contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    detected_ellipses = []
    
    for contour in contours:
        # We'll implement our own ellipse fitting algorithm here
        # 1. Calculate moments to find centroid
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        
        # Get centroid coordinates
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # 2. Get points of the contour
        points = contour.reshape(-1, 2)
        
        # 3. Calculate covariance matrix for points
        # This gives us information about the shape and orientation
        points = points.astype(np.float64)
        points_centered = points - np.array([cx, cy])
        
        # Skip if too few points
        if len(points_centered) < 5:
            continue
            
        # Calculate covariance matrix
        cov = np.cov(points_centered.T)
        
        # 4. Get eigenvalues and eigenvectors of the covariance matrix
        # These give us the axes and orientation of the ellipse
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # Sort eigenvalues and corresponding eigenvectors
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 5. Calculate major and minor axes
        # The eigenvalues are related to the variance along the axes
        major_axis = 2 * np.sqrt(5.991 * eigenvalues[0])  # 5.991 is chi-square value for 95% confidence
        minor_axis = 2 * np.sqrt(5.991 * eigenvalues[1])
        
        # Make sure we capture full extent of the blob
        # Find maximum distance from center to any contour point
        max_distance = 0
        for point in points:
            distance = np.sqrt((point[0] - cx)**2 + (point[1] - cy)**2)
            max_distance = max(max_distance, distance)
        
        # Adjust axes if needed
        scale_factor = 1.2 * max_distance / (major_axis / 2)
        major_axis *= scale_factor
        minor_axis *= scale_factor
        
        # 6. Calculate angle of orientation
        angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]) * 180 / np.pi
        
        # Format as OpenCV ellipse: ((center_x, center_y), (major_axis, minor_axis), angle)
        ellipse = ((cx, cy), (major_axis, minor_axis), angle)
        detected_ellipses.append(ellipse)
    
    print(f"Detected {len(detected_ellipses)} ellipses")
    return detected_ellipses


def draw_detected_shapes(result, shapes, line_color=(0, 255, 0), 
                        circle_color=(0, 0, 255), ellipse_color=(255, 0, 0), 
                        line_thickness=1):
    """
    Draw detected shapes on the input image.
    
    Returns:
     Image with shapes drawn
    """
    
    # Convert to color image if it's grayscale
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    # Draw lines
    if 'lines' in shapes:
        height, width = result.shape[:2]
        for rho, theta, _ in shapes['lines']:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            
            # Calculate endpoints of the line segment to draw
            # Extending the line to the image boundaries
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            
            # Clip line endpoints to image boundaries
            # Line equation: y = m*x + c
            if abs(a) < 1e-10:  # Vertical line
                x1 = x2 = int(rho)
                y1 = 0
                y2 = height - 1
            elif abs(b) < 1e-10:  # Horizontal line
                y1 = y2 = int(rho)
                x1 = 0
                x2 = width - 1
            else:
                # Slope and intercept
                m = -a / b
                c = y0 - m * x0
                
                # Find intersections with image boundaries
                # Left boundary (x=0)
                y_left = int(c)
                if 0 <= y_left < height:
                    x1, y1 = 0, y_left
                
                # Right boundary (x=width-1)
                y_right = int(m * (width - 1) + c)
                if 0 <= y_right < height:
                    x2, y2 = width - 1, y_right
                
                # Top boundary (y=0)
                x_top = int(-c / m) if abs(m) > 1e-10 else 0
                if 0 <= x_top < width:
                    if y1 < 0 or y1 >= height:
                        x1, y1 = x_top, 0
                    elif y2 < 0 or y2 >= height:
                        x2, y2 = x_top, 0
                
                # Bottom boundary (y=height-1)
                x_bottom = int((height - 1 - c) / m) if abs(m) > 1e-10 else 0
                if 0 <= x_bottom < width:
                    if y1 < 0 or y1 >= height:
                        x1, y1 = x_bottom, height - 1
                    elif y2 < 0 or y2 >= height:
                        x2, y2 = x_bottom, height - 1
            
            # Draw the line
            cv2.line(result, (x1, y1), (x2, y2), line_color, line_thickness)
    
    # Draw circles
    if 'circles' in shapes:
        for center_x, center_y, radius, _ in shapes['circles']:
            # Ensure values are integers
            center = (int(center_x), int(center_y))
            radius = int(radius)
            
            # Draw the circle
            cv2.circle(result, center, radius, circle_color, line_thickness)
            # Draw center point
            cv2.circle(result, center, 2, circle_color, -1)  # Filled circle for center
    
    # # Draw ellipses
    # if 'ellipses' in shapes:
    #     for center_x, center_y, a, b, angle, _ in shapes['ellipses']:
    #         # Ensure values are integers
    #         center = (int(center_x), int(center_y))
    #         axes = (int(a), int(b))
    #         angle = int(angle)
            
    #         # Draw the ellipse
    #         cv2.ellipse(result, center, axes, angle, 0, 360, ellipse_color, line_thickness)
    #         # Draw center point
    #         cv2.circle(result, center, 2, ellipse_color, -1)  # Filled circle for center

        # Draw ellipses
    if 'ellipses' in shapes:
        for ellipse in shapes['ellipses']:
            # OpenCV ellipse format: ((center_x, center_y), (major_axis, minor_axis), angle)
            center = (int(ellipse[0][0]), int(ellipse[0][1]))
            axes = (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2))  # Convert diameters to radii
            angle = int(ellipse[2])
            
            # Draw the ellipse
            cv2.ellipse(result, center, axes, angle, 0, 360, ellipse_color, line_thickness)
            # Draw center point
            cv2.circle(result, center, 2, ellipse_color, -1)  # Filled circle for center
    
    return result