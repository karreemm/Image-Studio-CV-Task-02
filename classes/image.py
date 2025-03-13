import cv2
import numpy as np
import random
from PyQt5.QtWidgets import QFileDialog


class Image():
    def __init__(self):
        self.input_image = None # kept untouched
        self.output_image = None # the one that will be modified by noise, filters etc.
        self.image_type = None # 'grey' or 'color' image
        self.input_image_fft = None # input image in frequency domain
        self.output_image_fft = None # output image in frequency domain

    def select_image(self):
        '''
        function to select and upload an image 
        '''
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.jpg *.jpeg *.png *.bmp *.gif *.tif)", options=options)
        if file_path:
            self.input_image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            self.output_image = self.input_image.copy() # a copy of the selected image is made so we can modify it without affecting the original image
            self.update_image_type(self.input_image) # update the selected image type
            self.input_image_fft , _ = self.fourier_transform(self.input_image)

    def update_image_type(self, image):
        '''
        function that detects whether image is grey or color (rgb) and updates the image_type attribute
        '''
        # splitting the image into its 3 color channels
        r,g,b = cv2.split(image)

        # getting differences between them
        r_g = np.count_nonzero(abs(r-g))
        r_b = np.count_nonzero(abs(r-b))
        g_b = np.count_nonzero(abs(g-b))
        diff_sum = float(r_g+r_b+g_b)

        # finding ratio of diff_sum with respect to size of image
        ratio = diff_sum/image.size
        if ratio > 0.005:
            self.image_type = 'color'
        else:
            self.image_type = 'grey'
    
    def apply_noise(self, noise_type , mean = 0 , sigma = np.sqrt(0.1)):
        if noise_type == 'uniform':
            self.output_image = self.output_image / 255.0
            x, y, dimensions = self.output_image.shape

            min = 0.0
            max = 0.2

            noise = np.zeros((x, y), dtype = np.float64)
            for i in range(x):
                for j in range(y):
                    noise[i][j] = random.uniform(min, max)

            r, g, b = cv2.split(self.output_image)
            r = r + noise
            g = g + noise
            b = b + noise
            self.output_image = cv2.merge((r, g, b))
            self.output_image = np.clip(self.output_image, 0.0, 1.0)
            self.output_image = (self.output_image * 255).astype(np.uint8)

        elif noise_type == 'gaussian':
            self.output_image = self.output_image / 255.0
            x, y, dimensions = self.output_image.shape
            # mean = 0
            # variance = 0.1
            # sigma = np.sqrt(variance) # standard deviation
            noise = np.random.normal(loc = mean,
                                 scale = sigma, # standard deviation
                                 size = (x, y))
            r, g, b = cv2.split(self.output_image)
            r = r + noise
            g = g + noise
            b = b + noise
            self.output_image = cv2.merge((r, g, b))
            self.output_image = np.clip(self.output_image, 0.0, 1.0)
            self.output_image = (self.output_image * 255).astype(np.uint8)

        # note: salt & pepper type is found only in grayscale images, that's why we convert the image to grayscale first before adding the noise
        elif noise_type == 'salt_pepper':
            self.output_image = self.convert_rgb_to_gray(self.output_image)
            channels = 1 if len(self.output_image) == 2 else self.output_image.shape[2]

            # Getting the dimensions of the image 
            row, col ,dimensions = self.output_image.shape 
            
            # randomly pick number of pixels (between 300 and 10000 pixls) in the image to be colored white 
            number_of_pixels = random.randint(300, 10000) 
            for c in range(channels):
                # add the salt and pepper noise at random positions in the image
                for i in range(number_of_pixels): 
                    
                    # a random y coordinate 
                    y_coord=random.randint(0, row - 1) 
                    
                    # a random x coordinate 
                    x_coord=random.randint(0, col - 1) 
                    
                    # color that pixel to white 
                    self.output_image[y_coord][x_coord][c] = 255
                    
                # same as above but for black pixels
                number_of_pixels = random.randint(300 , 10000) 
                for i in range(number_of_pixels): 
                    
                    y_coord=random.randint(0, row - 1) 
                    
                    x_coord=random.randint(0, col - 1) 
                    
                    # color that pixel to black
                    self.output_image[y_coord][x_coord][c] = 0
    
    def convert_rgb_to_gray(self, image):
        r,g,b = cv2.split(image)
        r = 0.299 * r
        g = 0.587 * g
        b = 0.114 * b
        grey_image = r + g + b
        grey_image = grey_image.astype(np.uint8)
        self.image_type = 'grey'

        grey_image_as_3_channels = cv2.merge((grey_image, grey_image, grey_image))   

        return grey_image_as_3_channels
    
    def fourier_transform(self, image):
        '''
        this function returns:
        
        1. shifted_frequency_domain_image -> which is the one to be passed on to the inverse_fourier_transform function to produce the time domain image.

        2. scaled_magnitude_image -> this one to be displayed in the GUI as the image in the frequency domain.

        Note: to do the fourier transform, the image must be in grayscale. that's why we convert it to grayscale directly in the first step.
        '''
        image_as_3_channels = self.convert_rgb_to_gray(image)

        image = image_as_3_channels[:,:,0] # we only need one channel since the image is in grayscale

        # compute the discrete Fourier Transform of the image. cv2.dft returns the Fourier Transform as a NumPy array.
        frequency_domain_image = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
        
        # Shift the zero-frequency component of the Fourier Transform to the center of the array because the cv2.dft() function returns the Fourier Transform with the zero-frequency component at the top-left corner of the array
        shifted_frequency_domain_image = np.fft.fftshift(frequency_domain_image)
        
        # calculate the magnitude and then take log and multiply by 20 to convert to dB
        magnitude_of_frequency_domain_image = 20*np.log(cv2.magnitude(shifted_frequency_domain_image[:,:,0],shifted_frequency_domain_image[:,:,1]))
        
        # scale the magnitude of the Fourier Transform using the cv2.normalize() function for improving the contrast of the resulting image
        scaled_magnitude_image = cv2.normalize(magnitude_of_frequency_domain_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        return shifted_frequency_domain_image, scaled_magnitude_image
    
    def inverse_fourier_transform(self , frequency_domain_image):
        '''
        when going from frequency domain to time domain, we need to reverse the process we did when going from time domain to frequency domain.
        '''
        shifted_frequency_image = np.fft.ifftshift(frequency_domain_image)
        time_domain_image = cv2.idft(shifted_frequency_image)
        time_domain_image = cv2.magnitude(time_domain_image[:,:,0], time_domain_image[:,:,1])
        time_domain_image = cv2.normalize(time_domain_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        time_domain_image = cv2.merge([time_domain_image, time_domain_image, time_domain_image])
        return time_domain_image # Note: returns 3 channel image although it is grayscale 
    
    def apply_filter(self, filter_type, filter_size, sigma = 1):
        ''' 
        apply low or high pass filter to image in the spatial domain
        we choose 3x3 kernel for each filter type
        '''
        if filter_type == 'Average':
            kernel = np.ones((filter_size,filter_size),np.float32)/9
            
        elif filter_type == 'Gaussian':
            x, y = np.meshgrid(np.arange(-filter_size // 2,(filter_size // 2 )+1), np.arange(-filter_size // 2,(filter_size // 2 )+1))  
            kernel = np.exp(-(x**2 + y**2)/(2*sigma**2))/(2*np.pi*sigma**2) 
            kernel = kernel / np.sum(kernel)    # normalization for the kernel
            
        elif filter_type == 'Median':
            padded_image = np.pad(self.output_image, pad_width=((filter_size // 2, filter_size // 2) , (filter_size // 2, filter_size // 2), (0, 0)), mode='constant', constant_values=0)  # pad the image with frame of zeros
            for i in range(self.output_image.shape[0]):
                for j in range(self.output_image.shape[1]):
                    self.output_image[i,j] = np.median(padded_image[i:i + filter_size, j:j + filter_size])
            return
        
        elif filter_type == 'Sobel':
            self.output_image = self.output_image.astype(np.float32)
            if filter_size == 3:
                horizontal_gradient = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] , dtype=np.float32)
                vertical_gradient = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]] , dtype=np.float32)
            elif filter_size == 5:
                horizontal_gradient = np.array([[2, 2, 4, 2, 2], [1, 1, 2, 1, 1], [0, 0, 0, 0, 0], [-1, -1, -2, -1, -1], [-2, -2, -4, -2, -2]] , dtype=np.float32)
                vertical_gradient = np.array([[2, 1, 0, -1, -2], [2, 1, 0, -1, -2], [4, 2, 0, -2, -4], [2, 1, 0, -1, -2], [2, 1, 0, -1, -2]] , dtype=np.float32)
            vertical_edges = self.convolve(self.output_image, horizontal_gradient)
            horizontal_edges = self.convolve(self.output_image, vertical_gradient)
            self.output_image = np.sqrt(vertical_edges**2 + horizontal_edges**2)
            self.output_image = self.normalize_image(self.output_image)
            return
        
        elif filter_type == 'Roberts':
            self.output_image = self.output_image.astype(np.float32)
            if filter_size == 3:
                vertical_gradient = np.array([[0, 0, 1], [0, -1, 0], [0, 0, 0]] , dtype=np.float32)
                horizontal_gradient = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]] , dtype= np.float32)
            elif filter_size == 5:
                vertical_gradient = np.array([[0, 0, 0, 0, 1], [0, 0, 0, 2, 0], [0, 0, 0, 0, 0], [0, -2, 0, 0, 0], [-1, 0, 0, 0, 0]] , dtype=np.float32)
                horizontal_gradient = np.array([[1, 0, 0, 0, 0], [0, 2, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, -2, 0], [0, 0, 0, 0, -1]] , dtype= np.float32)
            vertical_edges = self.convolve(self.output_image, horizontal_gradient)
            horizontal_edges = self.convolve(self.output_image, vertical_gradient)
            self.output_image = np.sqrt(vertical_edges**2 + horizontal_edges**2)
            self.output_image = self.normalize_image(self.output_image)
            return
        
        elif filter_type == 'Prewitt':
            self.output_image = self.output_image.astype(np.float32)
            if filter_size == 3:
                vertical_gradient = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
                horizontal_gradient = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            elif filter_size == 5:
                vertical_gradient = np.array([[-2, -2, -2, -2, -2], [-1, -1, -1, -1, -1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [2, 2, 2, 2, 2]])
                horizontal_gradient = np.array([[-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2]])
            vertical_edges = self.convolve(self.output_image, horizontal_gradient)
            horizontal_edges = self.convolve(self.output_image, vertical_gradient)
            self.output_image = np.sqrt(vertical_edges**2 + horizontal_edges**2)
            self.output_image = self.normalize_image(self.output_image)
            return
        elif filter_type == 'Canny':
            self.output_image = cv2.Canny(self.output_image, 100, 200)
            self.output_image = cv2.cvtColor(self.output_image, cv2.COLOR_GRAY2RGB)
            return
        
        self.output_image = self.convolve(self.output_image, kernel)
        self.output_image = np.clip(self.output_image, 0, 255).astype(np.uint8)    
    
    def frequency_domain_ideal_filter(self , type, shifted_fft_image, radius):
        '''
        low pass filter blurrs the image

        high pass filter acts like an edge detector
        '''
        rows, columns, dimensions = shifted_fft_image.shape

        if type == 'low':
            mask = np.zeros((rows, columns, 2), np.uint8)
            for u in range(rows):
                for v in range(columns):
                    D = np.sqrt((u - rows/2) ** 2 + (v - columns/2) ** 2) # D represents the (radius) distance from the center of the frequency domain
                    if D <= radius:
                        mask[u, v] = 1
                    else:
                        mask[u, v] = 0
        
        elif type == 'high':
            mask = np.ones((rows, columns, 2), np.uint8)
            for u in range(rows):
                for v in range(columns):
                    D = np.sqrt((u - rows/2) ** 2 + (v - columns/2) ** 2) # D represents the (radius) distance from the center of the frequency domain
                    if D >= radius:
                        mask[u, v] = 1
                    else:
                        mask[u, v] = 0

        filtered_fft_image = shifted_fft_image * mask
        return filtered_fft_image
    
    def frequency_domain_butterworth_filter(self, type, shifted_fft_image , D0 , n):
        rows, columns, dimensions = shifted_fft_image.shape
        H = np.zeros((rows, columns, 2), dtype=np.float32)
        
        if type == 'low':
            for u in range(rows):
                for v in range(columns):
                    D = np.sqrt((u - rows/2) ** 2 + (v - columns/2) ** 2) # D represents the (radius) distance from the center of the frequency domain
                    H[u, v] = 1 / (1 + (D / D0) ** (2 * n)) # H is the filter kernel/mask

        elif type == 'high':
            for u in range(rows):
                for v in range(columns):
                    D = np.sqrt((u - rows/2) ** 2 + (v - columns/2) ** 2) # D here represents same thing as other D
                    H[u, v] = 1 / (1 + (D0 / D) ** (2 * n)) # H is the filter kernel/mask

        filtered_image = shifted_fft_image * H 
        return filtered_image

    def frequency_domain_gaussian_filter(self, type, shifted_fft_image , D0):
        rows, columns, dimensions = shifted_fft_image.shape
        H = np.zeros((rows, columns, 2), dtype=np.float32)

        if type == 'low':
            for u in range(rows):
                for v in range(columns):
                    D = np.sqrt((u - rows/2)**2 + (v - columns/2)**2) # D represents the (radius) distance from the center of the frequency domain
                    H[u, v] = np.exp(-D**2 / (2 * D0 * D0)) # H is the filter kernel/mask
        
        elif type == 'high':
            '''
            the Gaussian high pass filter mask is simply 1 - the Gaussian low pass filter mask
            '''
            for u in range(rows):
                for v in range(columns):
                    D = np.sqrt((u - rows/2)**2 + (v - columns/2)**2)
                    H[u, v] = 1 - np.exp(-D**2 / (2 * D0 * D0)) 
        
        filtered_fft_image = shifted_fft_image * H
        return filtered_fft_image

    def apply_hybrid_image(self, image_1, image_2):
        '''
        NOTE: YOU MUST PASS TWO IMAGES OF THE SAME DIMENSIONS

        First argument is the low frequency components image
        Second argument is the high frequency components image
        '''
        # checking if the two images have the same dimensions
        if image_1.shape != image_2.shape:
            raise ValueError("Both images must have the same dimensions. "
                            f"Got {image_1.shape} and {image_2.shape}.")

        shifted_fft_image_1, _ = self.fourier_transform(image_1)
        shifted_fft_image_2, _ = self.fourier_transform(image_2)

        low_pass_filtered_image_1 = self.frequency_domain_gaussian_filter('low', shifted_fft_image_1 , 10)
        high_pass_filtered_image_2 = self.frequency_domain_gaussian_filter('high', shifted_fft_image_2 , 10)

        hybrid_image_fft = low_pass_filtered_image_1 + high_pass_filtered_image_2
        hybrid_image = self.inverse_fourier_transform(hybrid_image_fft)

        # time_domain_image_1 = self.inverse_fourier_transform(low_pass_filtered_image_1)
        # time_domain_image_2 = self.inverse_fourier_transform(high_pass_filtered_image_2)

        # hybrid_image = time_domain_image_1 + time_domain_image_2

        return hybrid_image
    
    def global_threshold(self, threshold_value=127):
        """
        Apply global thresholding to the image
        """
        # Convert to grayscale if the image is in color
        if self.image_type == 'color':
            grayscale_image = self.convert_rgb_to_gray(self.output_image)
            # Extract one channel since they're all the same in grayscale
            gray = grayscale_image[:,:,0]
        else:
            # For grayscale images, just take one channel
            gray = self.output_image[:,:,0]
        
        # Create a binary image
        binary_image = np.zeros_like(gray)
        
        # Apply thresholding manually
        height, width = gray.shape
        for i in range(height):
            for j in range(width):
                if gray[i, j] > threshold_value:
                    binary_image[i, j] = 255
                else:
                    binary_image[i, j] = 0
        
        # Convert back to 3-channel image
        self.output_image = cv2.merge([binary_image, binary_image, binary_image])

    def local_threshold(self, block_size=11, c=0):
        """
        Apply (local) thresholding to the image manually.
        """
        # Ensure block_size is odd
        if block_size % 2 == 0:
            block_size += 1
            
        # Convert to grayscale if the image is in color
        if self.image_type == 'color':
            grayscale_image = self.convert_rgb_to_gray(self.output_image)
            # Extract one channel since they're all the same in grayscale
            gray = grayscale_image[:,:,0]
        else:
            # For grayscale images, just take one channel
            gray = self.output_image[:,:,0]
            
        # Create output binary image
        height, width = gray.shape
        binary_image = np.zeros_like(gray)
            
        # Calculate the offset 
        offset = int(block_size // 2)
            
        # Apply padding to handle border pixels
        padded_image = np.pad(gray, ((offset, offset), (offset, offset)), mode='reflect')
            
        # Apply local thresholding 
        for i in range(height):
            for j in range(width):
                # Extract neighborhood - including the offset to center it
                i_padded = i + offset
                j_padded = j + offset
                neighborhood = padded_image[i_padded-offset:i_padded+offset+1, j_padded-offset:j_padded+offset+1]
                
                # Calculate local threshold (mean of neighborhood - c)
                local_threshold = np.mean(neighborhood) - c
                
                # Apply threshold
                if gray[i, j] > local_threshold:
                    binary_image[i, j] = 255
                else:
                    binary_image[i, j] = 0
            
        # Convert back to 3-channel image for consistency
        self.output_image = cv2.merge([binary_image, binary_image, binary_image])

    def convolve(self, image, kernel):
        '''
        implement convolution operation on the image 
        '''
        taken_padded_region = kernel.shape[0]
        image = image.astype(np.float32)
        channels = 1 if len(image.shape) == 2 else image.shape[2]
        padded_image = np.pad(image, pad_width=((kernel.shape[0] // 2, kernel.shape[0] // 2) , (kernel.shape[0] // 2, kernel.shape[0] // 2), (0, 0)), mode='constant', constant_values=0) # pad the image with frame of zeros
        
        output_image = np.zeros_like(image, dtype=np.float32)
        
        for c in range(channels):
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    region = padded_image[i:i+taken_padded_region, j:j+taken_padded_region, c]
                    output_image[i, j, c] = np.sum(region * kernel)
                    
        # output_image = np.clip(output_image, 0, 255).astype(np.uint8)
        return output_image
    
    def normalize_image(self , image, new_min=0, new_max=255):
        
        image = image.astype(np.float32) 

        old_min = np.min(image, axis=(0, 1), keepdims=True)
        old_max = np.max(image, axis=(0, 1), keepdims=True)

        scale = (new_max - new_min) / (old_max - old_min + 1e-8)
        normalized = (image - old_min) * scale + new_min

        return np.clip(normalized, new_min, new_max).astype(np.uint8)
    
    def histogram_equalization_channel(self, channel):
        histogram = np.zeros(256)
        for pixel in channel.ravel():
            histogram[pixel] += 1

        pdf = histogram / histogram.sum()
        cdf = np.cumsum(pdf)

        cdf_normalized = np.round(cdf * 255).astype(np.uint8)

        equalized_channel = cdf_normalized[channel]

        return equalized_channel

    def histogram_equalization_color(self , image):
        b, g, r = cv2.split(image)

        b_eq = self.histogram_equalization_channel(b)
        g_eq = self.histogram_equalization_channel(g)
        r_eq = self.histogram_equalization_channel(r)

        equalized_image = cv2.merge((b_eq, g_eq, r_eq))

        self.output_image = equalized_image