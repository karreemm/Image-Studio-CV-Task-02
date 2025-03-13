import cv2
from PyQt5.QtGui import QPixmap , QImage
from classes.enums import ContourMode
from copy import deepcopy


class Controller():
    def __init__(self , input_image , output_image ,contour_drawing_widget , output_image_label):
        self.input_image = input_image
        self.output_image = output_image
        self.contour_drawing_widget = contour_drawing_widget
        self.output_image_label = output_image_label
    
    def browse_input_image(self):
        self.input_image.select_image()
        self.output_image = deepcopy(self.input_image)
        self.output_image_label.setPixmap(self.numpy_to_qpixmap(self.output_image.input_image))
        self.output_image_label.setScaledContents(True)
    
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