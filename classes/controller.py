import cv2
from PyQt5.QtGui import QPixmap , QImage
from classes.enums import ContourMode
from classes.snake import Snake
from copy import deepcopy


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
        self.output_image_label.setPixmap(self.numpy_to_qpixmap(self.output_image.input_image))
        self.output_image_label.setScaledContents(True)
    
    def update_contour_drawing_widget_params(self):
        self.contour_drawing_widget.pixmap = self.numpy_to_qpixmap(self.input_image.input_image)
        self.contour_drawing_widget.current_mode = ContourMode.RECTANGLE
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
    
    def apply_canny_edge_detection(self, sigma, low_threshold, high_threshold):
        from classes.canny import apply_canny_edge_detection
        
        if self.input_image.input_image is not None:
            edges = apply_canny_edge_detection(self.input_image.input_image, sigma, low_threshold, high_threshold)
            
            # Convert to RGB if grayscale
            if len(edges.shape) == 2:
                edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            else:
                edges_rgb = edges
            
            # Update the output image
            self.output_image.input_image = edges_rgb
            self.output_image.output_image = edges_rgb
            
            # Update the display
            self.output_image_label.setPixmap(self.numpy_to_qpixmap(edges_rgb))
            self.output_image_label.setScaledContents(True)
        
        return edges_rgb
    
    def apply_snake_greedy(self):
        initial_contour_points = self.contour_drawing_widget.contour_points
        self.snake.convert_qpoints_to_list(initial_contour_points)
        new_contour_list = self.snake.greedy_snake(self.input_image.input_image , self.snake.contour_points)
        new_contour_qpoints = self.snake.convert_list_to_qpoints(new_contour_list)
        self.contour_drawing_widget.contour_points = new_contour_qpoints
        self.contour_drawing_widget.update()