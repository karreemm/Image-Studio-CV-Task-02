import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow , QPushButton , QStackedWidget ,QFrame , QLabel , QVBoxLayout, QHBoxLayout , QCheckBox , QComboBox , QLineEdit
from PyQt5.uic import loadUi
from PyQt5.QtGui import QIcon
import cv2
from helper_functions.compile_qrc import compile_qrc
from classes.image import Image
from classes.controller import Controller
from classes.enums import ImageSource , Channel
from classes.contourDrawingWidget import ContourDrawingWidget

compile_qrc()
from icons_setup.icons import *

from icons_setup.compiledIcons import *
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi('main.ui', self)
        self.setWindowTitle('Image Studio')
        self.setWindowIcon(QIcon('icons_setup\icons\logo.png'))
        
        # Initialize the Input Image Frame
        self.input_image_frame = self.findChild(QFrame , "inputFrame")

        # Initialize Contour Drawing Widget
        self.contour_drawing_widget = ContourDrawingWidget(self.input_image_frame)
        
        # Initialize Input Image Layout
        self.input_image_layout = QVBoxLayout(self.input_image_frame)
        
        self.input_image_layout.addWidget(self.contour_drawing_widget)
        self.input_image_frame.setLayout(self.input_image_layout)
        
        # Initialize Browse Image Button
        self.browse_image_button = self.findChild(QPushButton , "browse")
        self.browse_image_button.clicked.connect(self.browse_image)
        
        # Initialize Input Image
        self.input_image = Image()
        
        # Initialize Output Image
        self.output_image = Image()
        
        # Initialize Output Image Frame
        self.output_image_frame = self.findChild(QFrame , "outputFrame")
        self.output_image_label = QLabel(self.output_image_frame)
        self.output_image_layout = QVBoxLayout(self.output_image_frame)
        self.output_image_layout.addWidget(self.output_image_label)
        self.output_image_frame.setLayout(self.output_image_layout)

        # Initialize Canny Edge Detection Input Fields
        self.sigma_input = self.findChild(QLineEdit, "cannySigmaInput")
        self.sigma_input.setText("1")
        self.t_low_input = self.findChild(QLineEdit, "cannyTLowInput")
        self.t_low_input.setText("40")
        self.t_high_input = self.findChild(QLineEdit, "cannyTHighInput")
        self.t_high_input.setText("100")

        # Initialize Apply Canny Button
        self.cannyApplyButton = self.findChild(QPushButton, "cannyApplyButton")
        self.cannyApplyButton.clicked.connect(self.apply_canny_edge_detection)

        # Initialize Reset Button
        self.reset_button = self.findChild(QPushButton, "reset")
        self.reset_button.clicked.connect(self.reset_image)
        
        # Initialize Apply Snake Button
        self.apply_snake_greedy_button = self.findChild(QPushButton , "snakeApplyButton")
        self.apply_snake_greedy_button.clicked.connect(self.apply_snake_greedy)
        
        # Initialize Controller
        self.controller = Controller(self.input_image , self.output_image ,
                                    self.contour_drawing_widget , self.output_image_label)
        
    def browse_image(self):
        self.controller.browse_input_image()
        self.controller.update_contour_drawing_widget_params()

    def reset_image(self):
        self.sigma_input.setText("1")
        self.t_low_input.setText("40")
        self.t_high_input.setText("100")

        # Update the output image display
        self.output_image_label.setPixmap(self.controller.numpy_to_qpixmap(self.input_image.input_image))
        self.output_image_label.setScaledContents(True)

    def apply_snake_greedy(self):
        self.controller.apply_snake_greedy()
    
    def apply_canny_edge_detection(self):
        
        # Default values if fields are empty
        sigma = 1.4
        t_low = 40
        t_high = 100
        
        # Try to convert inputs to appropriate values
        try:
            if self.sigma_input.text():
                sigma = float(self.sigma_input.text())
            if self.t_low_input.text():
                t_low = float(self.t_low_input.text())
            if self.t_high_input.text():
                t_high = float(self.t_high_input.text())
        except ValueError:
            # If conversion fails, use default values
            pass
        
        # Apply Canny edge detection
        if self.input_image.input_image is not None:
            from classes.canny import apply_canny_edge_detection
            
            # Apply the Canny edge detection
            edges = apply_canny_edge_detection(self.input_image.input_image, sigma, t_low, t_high)
            
            # Convert the edges to 3-channel RGB for display
            if len(edges.shape) == 2:  # If single channel
                edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            else:
                edges_rgb = edges
            
            # Update the output image
            self.output_image.input_image = edges_rgb
            self.output_image.output_image = edges_rgb
            
            # Update the output image display
            self.output_image_label.setPixmap(self.controller.numpy_to_qpixmap(edges_rgb))
            self.output_image_label.setScaledContents(True)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())