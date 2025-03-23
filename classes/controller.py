import cv2
from PyQt5.QtGui import QPixmap , QImage
from classes.enums import ContourMode
from classes.snake import Snake
from copy import deepcopy
from classes.canny import convert_rgb_to_gray
from classes.image import Image
from classes.canny import apply_canny_edge_detection, detect_shapes, draw_detected_shapes

class Controller():
    def __init__(self , input_image , output_image ,contour_drawing_widget , output_image_label):
        self.input_image = input_image
        self.output_image = output_image
        self.contour_drawing_widget = contour_drawing_widget
        self.output_image_label = output_image_label
        self.snake = Snake()
    
    def browse_input_image(self):
        self.input_image.select_image()
        self.output_image = deepcopy(self.input_image)
        self.output_image_label.pixmap = self.numpy_to_qpixmap(self.output_image.input_image)
        self.output_image_label.setPixmap(self.output_image_label.pixmap)
        self.output_image_label.setScaledContents(True)
        self.output_image_label.contour_points = []
        self.output_image_label.image = self.output_image_label.pixmap.copy()
        self.output_image_label.setDrawingEnabled(False)
        self.output_image_label.update()
    
    def update_contour_drawing_widget_params(self):
        self.contour_drawing_widget.pixmap = self.numpy_to_qpixmap(self.input_image.input_image)
        self.contour_drawing_widget.current_mode = ContourMode.FREE
        self.contour_drawing_widget.start_point = None
        self.contour_drawing_widget.end_point = None
        self.contour_drawing_widget.contour_points = []  
        self.contour_drawing_widget.setPixmap(self.contour_drawing_widget.pixmap)
        self.contour_drawing_widget.setScaledContents(True)
        self.contour_drawing_widget.image = self.contour_drawing_widget.pixmap.copy()
        self.contour_drawing_widget.update()
        
    def numpy_to_qpixmap(self, image_array):
        """Convert NumPy array to QPixmap"""
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        height, width, channels = image_array.shape
        bytes_per_line = channels * width
        qimage = QImage(image_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qimage)
    
    # def apply_canny_edge_detection(self, sigma, low_threshold, high_threshold):
    #     from classes.canny import apply_canny_edge_detection
        
    #     if self.input_image.input_image is not None:
    #         edges = apply_canny_edge_detection(self.input_image.input_image, sigma, low_threshold, high_threshold)
            
    #         # Convert to RGB if grayscale
    #         if len(edges.shape) == 2:
    #             edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    #         else:
    #             edges_rgb = edges
            
    #         # Update the output image
    #         self.output_image.input_image = edges_rgb
    #         self.output_image.output_image = edges_rgb
            
    #         # Update the display
    #         self.output_image_label.setPixmap(self.numpy_to_qpixmap(edges_rgb))
    #         self.output_image_label.setScaledContents(True)
        
    #     return edges_rgb



    def apply_canny_edge_detection(self, sigma, low_threshold, high_threshold, 
                              detect_lines=False, detect_circles=False, detect_ellipses=False,
                              line_vote_threshold=50, circle_vote_threshold=50, ellipse_vote_threshold=50):
        if self.input_image.input_image is not None:
            # Step 1: Apply Canny edge detection
            edges = apply_canny_edge_detection(self.input_image.input_image, sigma, low_threshold, high_threshold)
            
            # Create a copy of the original image for drawing shapes
            # result_image = self.input_image.input_image.copy()
            result_image = edges.copy()
            
            # Step 2: Detect shapes if any options are selected
            if detect_lines or detect_circles or detect_ellipses:
                detected_shapes = detect_shapes(
                    self.input_image.input_image,
                    edges, 
                    detect_lines=detect_lines, 
                    detect_circles=detect_circles, 
                    detect_ellipses=detect_ellipses,
                    line_vote_threshold_percent=line_vote_threshold,
                    circle_vote_threshold_percent=circle_vote_threshold,
                    ellipse_vote_threshold_percent=ellipse_vote_threshold
                )
                
                # Step 3: Draw detected shapes on the result image
                result_image = draw_detected_shapes(result_image, detected_shapes)
            else:
                # If no shape detection is requested, just use the edge detection result
                # Convert to RGB if grayscale
                if len(edges.shape) == 2:
                    result_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                else:
                    result_image = edges
            
            # Update the output image
            self.output_image.input_image = result_image
            self.output_image.output_image = result_image
                    
            # Update the display
            self.output_image_label.setPixmap(self.numpy_to_qpixmap(result_image))
            self.output_image_label.setScaledContents(True)
                
        return result_image



    def apply_snake_greedy(self ,alpha , beta , gamma ,window_size):
        initial_contour_points = self.contour_drawing_widget.contour_points
        self.snake.convert_qpoints_to_list(initial_contour_points)
        # thresholded_image = apply_canny_edge_detection(self.input_image.input_image , 1 ,40,150)
        greyed_image = convert_rgb_to_gray(self.input_image.input_image)
        # _ , thresholded = cv2.threshold(greyed_image , 127,255,cv2.THRESH_BINARY)
        new_contour_list = self.snake.active_contour_greedy( greyed_image, self.snake.contour_points , alpha= alpha , beta=beta , gamma= gamma , search_window_size= window_size)
        new_contour_qpoints = self.snake.convert_list_to_qpoints(new_contour_list)
        self.output_image_label.contour_points = new_contour_qpoints
        self.output_image_label.update()
        