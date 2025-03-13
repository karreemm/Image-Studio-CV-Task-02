from classes.enums import Channel , DistributionCurve
import cv2
import numpy as np
from PyQt5.QtGui import QPixmap , QImage
from copy import deepcopy
class ControllerOld():
    def __init__(self , input_image_1 , input_image_2 , output_image , output_image_label ,
                input_image_1_red_histogram_canvas,input_image_1_green_histogram_canvas,input_image_1_blue_histogram_canvas,
                input_image_1_red_cdf_canvas , input_image_1_green_cdf_canvas , input_image_1_blue_cdf_canvas,
                input_image_1_red_pdf_canvas,input_image_1_green_pdf_canvas,input_image_1_blue_pdf_canvas,
                input_image_2_red_histogram_canvas ,input_image_2_green_histogram_canvas, input_image_2_blue_histogram_canvas,
                input_image_2_red_cdf_canvas , input_image_2_green_cdf_canvas , input_image_2_blue_cdf_canvas,
                input_image_2_red_pdf_canvas,input_image_2_green_pdf_canvas,input_image_2_blue_pdf_canvas,
                output_image_red_histogram_canvas , output_image_green_histogram_canvas, output_image_blue_histogram_canvas,
                output_image_red_cdf_canvas, output_image_green_cdf_canvas , output_image_blue_cdf_canvas,
                output_image_red_pdf_canvas , output_image_green_pdf_canvas, output_image_blue_pdf_canvas,
                hybrid_image,
                low_freq_image , high_freq_image):
        
        self.input_image_1 = input_image_1
        self.input_image_2 = input_image_2
        self.output_image = output_image
        self.hybrid_image = hybrid_image
        self.current_output_source_index = 0
        self.output_image_label = output_image_label
        self.input_image_1_red_histogram_canvas = input_image_1_red_histogram_canvas
        self.input_image_1_green_histogram_canvas = input_image_1_green_histogram_canvas
        self.input_image_1_blue_histogram_canvas = input_image_1_blue_histogram_canvas
        self.input_image_1_red_cdf_canvas = input_image_1_red_cdf_canvas
        self.input_image_1_green_cdf_canvas = input_image_1_green_cdf_canvas
        self.input_image_1_blue_cdf_canvas = input_image_1_blue_cdf_canvas
        self.input_image_1_red_pdf_canvas = input_image_1_red_pdf_canvas
        self.input_image_1_green_pdf_canvas = input_image_1_green_pdf_canvas
        self.input_image_1_blue_pdf_canvas = input_image_1_blue_pdf_canvas
        self.input_image_2_red_histogram_canvas = input_image_2_red_histogram_canvas
        self.input_image_2_green_histogram_canvas = input_image_2_green_histogram_canvas
        self.input_image_2_blue_histogram_canvas = input_image_2_blue_histogram_canvas
        self.input_image_2_red_cdf_canvas = input_image_2_red_cdf_canvas
        self.input_image_2_green_cdf_canvas = input_image_2_green_cdf_canvas
        self.input_image_2_blue_cdf_canvas = input_image_2_blue_cdf_canvas
        self.input_image_2_red_pdf_canvas = input_image_2_red_pdf_canvas
        self.input_image_2_green_pdf_canvas = input_image_2_green_pdf_canvas
        self.input_image_2_blue_pdf_canvas = input_image_2_blue_pdf_canvas
        self.output_image_red_histogram_canvas = output_image_red_histogram_canvas
        self.output_image_green_histogram_canvas = output_image_green_histogram_canvas
        self.output_image_blue_histogram_canvas = output_image_blue_histogram_canvas
        self.output_image_red_cdf_canvas = output_image_red_cdf_canvas
        self.output_image_green_cdf_canvas = output_image_green_cdf_canvas
        self.output_image_blue_cdf_canvas = output_image_blue_cdf_canvas
        self.output_image_red_pdf_canvas = output_image_red_pdf_canvas
        self.output_image_green_pdf_canvas = output_image_green_pdf_canvas
        self.output_image_blue_pdf_canvas = output_image_blue_pdf_canvas
        self.hybrid_image_mode = False
        self.low_freq_image = low_freq_image
        self.high_freq_image = high_freq_image
        self.current_graph_channel = Channel.RED
        self.current_distribution_curve = DistributionCurve.CDF

    def set_output_image_source(self):
        if(self.hybrid_image_mode):
            self.output_image_pixmap = self.numpy_to_qpixmap(self.hybrid_image.output_image)
            self.output_image_red_histogram_canvas.plot_histogram(self.hybrid_image.output_image[:,:,Channel.RED.value])
            self.output_image_green_histogram_canvas.plot_histogram(self.hybrid_image.output_image[:,:,Channel.GREEN.value])
            self.output_image_blue_histogram_canvas.plot_histogram(self.hybrid_image.output_image[:,:,Channel.BLUE.value])
            self.output_image_blue_histogram_canvas.plot_histogram(self.hybrid_image.output_image[:,:,Channel.BLUE.value])
            self.output_image_red_cdf_canvas.plot_distribution_curve(self.hybrid_image.output_image[:,:,Channel.RED.value] , "CDF")
            self.output_image_green_cdf_canvas.plot_distribution_curve(self.hybrid_image.output_image[:,:,Channel.GREEN.value] , "CDF")
            self.output_image_blue_cdf_canvas.plot_distribution_curve(self.hybrid_image.output_image[:,:,Channel.BLUE.value] , "CDF")            
            self.output_image_red_pdf_canvas.plot_distribution_curve(self.hybrid_image.output_image[:,:,Channel.RED.value] , "PDF")
            self.output_image_green_pdf_canvas.plot_distribution_curve(self.hybrid_image.output_image[:,:,Channel.GREEN.value] , "PDF")
            self.output_image_blue_pdf_canvas.plot_distribution_curve(self.hybrid_image.output_image[:,:,Channel.BLUE.value] , "PDF")
            
        elif(self.current_output_source_index == 1 and len(self.input_image_1.output_image) != 0):
            self.output_image_pixmap = self.numpy_to_qpixmap(self.input_image_1.output_image)
            self.output_image_red_histogram_canvas.plot_histogram(self.input_image_1.output_image[:,:,Channel.RED.value])
            self.output_image_green_histogram_canvas.plot_histogram(self.input_image_1.output_image[:,:,Channel.GREEN.value])
            self.output_image_blue_histogram_canvas.plot_histogram(self.input_image_1.output_image[:,:,Channel.BLUE.value])
            self.output_image_red_cdf_canvas.plot_distribution_curve(self.input_image_1.output_image[:,:,Channel.RED.value] , "CDF")
            self.output_image_green_cdf_canvas.plot_distribution_curve(self.input_image_1.output_image[:,:,Channel.GREEN.value] , "CDF")
            self.output_image_blue_cdf_canvas.plot_distribution_curve(self.input_image_1.output_image[:,:,Channel.BLUE.value] , "CDF")            
            self.output_image_red_pdf_canvas.plot_distribution_curve(self.input_image_1.output_image[:,:,Channel.RED.value] , "PDF")
            self.output_image_green_pdf_canvas.plot_distribution_curve(self.input_image_1.output_image[:,:,Channel.GREEN.value] , "PDF")
            self.output_image_blue_pdf_canvas.plot_distribution_curve(self.input_image_1.output_image[:,:,Channel.BLUE.value] , "PDF")
        
        elif(self.current_output_source_index == 2 and len(self.input_image_2.output_image) != 0):
            self.output_image_pixmap = self.numpy_to_qpixmap(self.input_image_2.output_image)
            self.output_image_red_histogram_canvas.plot_histogram(self.input_image_2.output_image[:,:,Channel.RED.value])
            self.output_image_green_histogram_canvas.plot_histogram(self.input_image_2.output_image[:,:,Channel.GREEN.value])
            self.output_image_blue_histogram_canvas.plot_histogram(self.input_image_2.output_image[:,:,Channel.BLUE.value])
            self.output_image_red_cdf_canvas.plot_distribution_curve(self.input_image_2.output_image[:,:,Channel.RED.value] , "CDF")
            self.output_image_green_cdf_canvas.plot_distribution_curve(self.input_image_2.output_image[:,:,Channel.GREEN.value] , "CDF")
            self.output_image_blue_cdf_canvas.plot_distribution_curve(self.input_image_2.output_image[:,:,Channel.BLUE.value] , "CDF")            
            self.output_image_red_pdf_canvas.plot_distribution_curve(self.input_image_2.output_image[:,:,Channel.RED.value] , "PDF")
            self.output_image_green_pdf_canvas.plot_distribution_curve(self.input_image_2.output_image[:,:,Channel.GREEN.value] , "PDF")
            self.output_image_blue_pdf_canvas.plot_distribution_curve(self.input_image_2.output_image[:,:,Channel.BLUE.value] , "PDF")        
        if (self.input_image_1.input_image is not None):                
            self.input_image_1_red_histogram_canvas.plot_histogram(self.input_image_1.input_image[:,:,Channel.RED.value])
            self.input_image_1_green_histogram_canvas.plot_histogram(self.input_image_1.input_image[:,:,Channel.GREEN.value])
            self.input_image_1_blue_histogram_canvas.plot_histogram(self.input_image_1.input_image[:,:,Channel.BLUE.value])
            self.input_image_1_red_cdf_canvas.plot_distribution_curve(self.input_image_1.input_image[:,:,Channel.RED.value] , "CDF")
            self.input_image_1_green_cdf_canvas.plot_distribution_curve(self.input_image_1.input_image[:,:,Channel.GREEN.value] , "CDF")
            self.input_image_1_blue_cdf_canvas.plot_distribution_curve(self.input_image_1.input_image[:,:,Channel.BLUE.value] , "CDF")            
            self.input_image_1_red_pdf_canvas.plot_distribution_curve(self.input_image_1.input_image[:,:,Channel.RED.value] , "PDF")
            self.input_image_1_green_pdf_canvas.plot_distribution_curve(self.input_image_1.input_image[:,:,Channel.GREEN.value] , "PDF")
            self.input_image_1_blue_pdf_canvas.plot_distribution_curve(self.input_image_1.input_image[:,:,Channel.BLUE.value] , "PDF")
            
        if (self.input_image_2.input_image is not None):
            self.input_image_2_red_histogram_canvas.plot_histogram(self.input_image_2.input_image[:,:,Channel.RED.value])
            self.input_image_2_blue_histogram_canvas.plot_histogram(self.input_image_2.input_image[:,:,Channel.BLUE.value])
            self.input_image_2_green_histogram_canvas.plot_histogram(self.input_image_2.input_image[:,:,Channel.GREEN.value])
            self.input_image_2_red_cdf_canvas.plot_distribution_curve(self.input_image_2.input_image[:,:,Channel.RED.value] , "CDF")
            self.input_image_2_green_cdf_canvas.plot_distribution_curve(self.input_image_2.input_image[:,:,Channel.GREEN.value] , "CDF")
            self.input_image_2_blue_cdf_canvas.plot_distribution_curve(self.input_image_2.input_image[:,:,Channel.BLUE.value] , "CDF")            
            self.input_image_2_red_pdf_canvas.plot_distribution_curve(self.input_image_2.input_image[:,:,Channel.RED.value] , "PDF")
            self.input_image_2_green_pdf_canvas.plot_distribution_curve(self.input_image_2.input_image[:,:,Channel.GREEN.value] , "PDF")
            self.input_image_2_blue_pdf_canvas.plot_distribution_curve(self.input_image_2.input_image[:,:,Channel.BLUE.value] , "PDF")        
        
        self.output_image_label.setPixmap(self.output_image_pixmap)
        self.output_image_label.setScaledContents(True)
        
    def browse_image_input_1(self):
        self.input_image_1.select_image()
        self.current_output_source_index = 1
        self.input_image_1_red_histogram_canvas.plot_histogram(self.input_image_1.input_image[:,:,Channel.RED.value])
        self.input_image_1_green_histogram_canvas.plot_histogram(self.input_image_1.input_image[:,:,Channel.GREEN.value])
        self.input_image_1_blue_histogram_canvas.plot_histogram(self.input_image_1.input_image[:,:,Channel.BLUE.value])
        self.input_image_1_red_cdf_canvas.plot_distribution_curve(self.input_image_1.input_image[:,:,Channel.RED.value] , "CDF")
        self.input_image_1_green_cdf_canvas.plot_distribution_curve(self.input_image_1.input_image[:,:,Channel.GREEN.value] , "CDF")
        self.input_image_1_blue_cdf_canvas.plot_distribution_curve(self.input_image_1.input_image[:,:,Channel.BLUE.value] , "CDF")
        self.input_image_1_red_pdf_canvas.plot_distribution_curve(self.input_image_1.input_image[:,:,Channel.RED.value] , "PDF")
        self.input_image_1_green_pdf_canvas.plot_distribution_curve(self.input_image_1.input_image[:,:,Channel.GREEN.value] , "PDF")
        self.input_image_1_blue_pdf_canvas.plot_distribution_curve(self.input_image_1.input_image[:,:,Channel.BLUE.value] , "PDF")
    
    def browse_image_input_2(self):
        self.input_image_2.select_image()
        self.current_output_source_index = 2
        self.input_image_2_red_histogram_canvas.plot_histogram(self.input_image_2.input_image[:,:,Channel.RED.value])
        self.input_image_2_blue_histogram_canvas.plot_histogram(self.input_image_2.input_image[:,:,Channel.BLUE.value])
        self.input_image_2_green_histogram_canvas.plot_histogram(self.input_image_2.input_image[:,:,Channel.GREEN.value])
        self.input_image_2_red_cdf_canvas.plot_distribution_curve(self.input_image_2.input_image[:,:,Channel.RED.value] , "CDF")
        self.input_image_2_green_cdf_canvas.plot_distribution_curve(self.input_image_2.input_image[:,:,Channel.GREEN.value] , "CDF")
        self.input_image_2_blue_cdf_canvas.plot_distribution_curve(self.input_image_2.input_image[:,:,Channel.BLUE.value] , "CDF")            
        self.input_image_2_red_pdf_canvas.plot_distribution_curve(self.input_image_2.input_image[:,:,Channel.RED.value] , "PDF")
        self.input_image_2_green_pdf_canvas.plot_distribution_curve(self.input_image_2.input_image[:,:,Channel.GREEN.value] , "PDF")
        self.input_image_2_blue_pdf_canvas.plot_distribution_curve(self.input_image_2.input_image[:,:,Channel.BLUE.value] , "PDF")

    def apply_noise(self , noise_type, mean = 0, std = np.sqrt(0.1)):
        if(self.input_image_1.input_image is not None):
            self.input_image_1.apply_noise(noise_type, mean ,std)
        if(self.input_image_2.input_image is not None ):
            self.input_image_2.apply_noise(noise_type , mean , std)
        self.set_output_image_source()
        
        
    def reset_output_image_to_normal(self):
        self.input_image_1.output_image = deepcopy(self.input_image_1.input_image)
        self.input_image_2.output_image = deepcopy(self.input_image_2.input_image)
        self.set_output_image_source()

    def rgb2grey(self):
        if(self.input_image_1.input_image is not None):
            self.input_image_1.output_image = self.input_image_1.convert_rgb_to_gray(self.input_image_1.input_image)
        if(self.input_image_2.input_image is not None ):
            self.input_image_2.output_image = self.input_image_2.convert_rgb_to_gray(self.input_image_2.input_image)
        self.set_output_image_source()
    
    def apply_time_domain_low_pass(self , filter_type ,filter_size, sigma = 1):
        if(self.input_image_1.input_image is not None):
            self.input_image_1.apply_filter(filter_type ,filter_size, sigma)
        if(self.input_image_2.input_image is not None ):
            self.input_image_2.apply_filter(filter_type ,filter_size, sigma)
        self.set_output_image_source()
    
    def apply_edge_detector_time_domain(self , filter_type,filter_size):
        if(self.input_image_1.input_image is not None):
            self.input_image_1.apply_filter(filter_type,filter_size)
        if(self.input_image_2.input_image is not None ):
            self.input_image_2.apply_filter(filter_type,filter_size)
        self.set_output_image_source()

    def apply_hybrid_image(self , low_freq_image , high_freq_image):
        self.hybrid_image.output_image = self.hybrid_image.apply_hybrid_image(low_freq_image , high_freq_image)
        
    def numpy_to_qpixmap(self, image_array):
        """Convert NumPy array to QPixmap"""
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        height, width, channels = image_array.shape
        bytes_per_line = channels * width
        qimage = QImage(image_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qimage)
    
    def apply_local_thresholding(self , window_size , c):
        if(self.input_image_1.input_image is not None):
            self.input_image_1.local_threshold(window_size , c)
        if(self.input_image_2.input_image is not None ):
            self.input_image_2.local_threshold(window_size , c)
        self.set_output_image_source()

    def apply_global_thresholding(self , threshold):
        if(self.input_image_1.input_image is not None):
            self.input_image_1.global_threshold(threshold)
        if(self.input_image_2.input_image is not None ):
            self.input_image_2.global_threshold(threshold)
        self.set_output_image_source()
        
    def apply_histogram_equalization(self):
        if(self.current_output_source_index == 1 and len(self.input_image_1.output_image) != 0 ):
            self.input_image_1.histogram_equalization_color(self.input_image_1.input_image)
        
        if(self.current_output_source_index == 2 and len(self.input_image_2.output_image) != 0 ):
            self.input_image_2.histogram_equalization_color(self.input_image_2.input_image)
        
        self.set_output_image_source()

    def apply_ideal_freq_filters(self , type ,radius):
        if(self.current_output_source_index == 1 and len(self.input_image_1.output_image) != 0 ):
            output_image1_fft = self.input_image_1.frequency_domain_ideal_filter(type , self.input_image_1.input_image_fft , radius)
            self.input_image_1.output_image = self.input_image_1.inverse_fourier_transform(output_image1_fft)
        if(self.current_output_source_index == 2 and len(self.input_image_2.output_image) != 0 ):
            output_image2_fft = self.input_image_2.frequency_domain_ideal_filter(type , self.input_image_2.input_image_fft , radius)
            self.input_image_2.output_image = self.input_image_2.inverse_fourier_transform(output_image2_fft)
        self.set_output_image_source()
    
    def apply_butter_freq_filters(self , type , cutoff , order):
        if(self.current_output_source_index == 1 and len(self.input_image_1.output_image) != 0 ):
            output_image1_fft = self.input_image_1.frequency_domain_butterworth_filter(type , self.input_image_1.input_image_fft , cutoff , order)
            self.input_image_1.output_image = self.input_image_1.inverse_fourier_transform(output_image1_fft)
        if(self.current_output_source_index == 2 and len(self.input_image_2.output_image) != 0 ):
            output_image2_fft = self.input_image_2.frequency_domain_butterworth_filter(type , self.input_image_2.input_image_fft , cutoff , order)
            self.input_image_2.output_image = self.input_image_2.inverse_fourier_transform(output_image2_fft)
        self.set_output_image_source()
    
    def apply_gaussian_freq_filters(self , type , cutoff ):
        if(self.current_output_source_index == 1 and len(self.input_image_1.output_image) != 0 ):
            output_image1_fft = self.input_image_1.frequency_domain_gaussian_filter(type , self.input_image_1.input_image_fft , cutoff)
            self.input_image_1.output_image = self.input_image_1.inverse_fourier_transform(output_image1_fft)
        if(self.current_output_source_index == 2 and len(self.input_image_2.output_image) != 0 ):
            output_image2_fft = self.input_image_2.frequency_domain_gaussian_filter(type , self.input_image_2.input_image_fft , cutoff)
            self.input_image_2.output_image = self.input_image_2.inverse_fourier_transform(output_image2_fft)
        self.set_output_image_source()