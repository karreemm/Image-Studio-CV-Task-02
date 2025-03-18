import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow , QPushButton , QStackedWidget ,QFrame , QLabel , QVBoxLayout, QHBoxLayout , QCheckBox , QComboBox , QLineEdit, QSlider
from PyQt5.uic import loadUi
from PyQt5.QtGui import QIcon
import cv2
from helper_functions.compile_qrc import compile_qrc
from classes.image import Image
from classes.controller import Controller
from classes.enums import ContourMode
from classes.contourDrawingWidget import ContourDrawingWidget
from classes.snake import Snake
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
        self.output_contour_drawing = ContourDrawingWidget(self.output_image_frame)
        self.output_image_layout = QVBoxLayout(self.output_image_frame)
        self.output_image_layout.addWidget(self.output_contour_drawing)
        self.output_image_frame.setLayout(self.output_image_layout)

        # Initialize Canny Edge Detection Input Fields
        self.sigma_input = self.findChild(QLineEdit, "cannySigmaInput")
        self.sigma_input.setText("1")
        self.sigma_input.textChanged.connect(self.update_sigma)
        self.t_low_input = self.findChild(QLineEdit, "cannyTLowInput")
        self.t_low_input.setText("40")
        self.t_low_input.textChanged.connect(self.update_t_low)
        self.t_high_input = self.findChild(QLineEdit, "cannyTHighInput")
        self.t_high_input.setText("100")
        self.t_high_input.textChanged.connect(self.update_t_high)

        # Initialize Apply Canny Button
        self.cannyApplyButton = self.findChild(QPushButton, "cannyApplyButton")
        self.cannyApplyButton.clicked.connect(self.apply_canny_edge_detection)

        # Initialize % of max votes of shapes in accumulator
        self.cannyMaxVoteLinesValue = 50
        self.cannyMaxVoteCirclesValue = 50
        self.cannyMaxVoteEllipsesValue = 50

        self.cannyMaxVoteEllipses = self.findChild(QLabel, "cannyMaxVoteEllipses")
        self.cannyMaxVoteCircles = self.findChild(QLabel, "cannyMaxVoteCircles")
        self.cannyMaxVoteLines = self.findChild(QLabel, "cannyMaxVoteLines")

        self.cannyMaxVoteEllipsesSlider = self.findChild(QSlider, "cannyMaxVoteEllipsesSlider")
        self.cannyMaxVoteCirclesSlider = self.findChild(QSlider, "cannyMaxVoteCirclesSlider")
        self.cannyMaxVoteLinesSlider = self.findChild(QSlider, "cannyMaxVoteLinesSlider")

        for slider in [self.cannyMaxVoteEllipsesSlider, self.cannyMaxVoteCirclesSlider, self.cannyMaxVoteLinesSlider]:
            slider.setMinimum(0)
            slider.setMaximum(100)
            slider.setValue(50)

        self.cannyMaxVoteEllipsesSlider.valueChanged.connect(self.update_canny_max_vote_ellipses)
        self.cannyMaxVoteCirclesSlider.valueChanged.connect(self.update_canny_max_vote_circles)
        self.cannyMaxVoteLinesSlider.valueChanged.connect(self.update_canny_max_vote_lines)
        

        # Initialize Shape Detection Checkboxes
        self.isCannyLinesDetected = False
        self.isCannyCirclesDetected = False
        self.isCannyEllipsesDetected = False

        self.detectEllipsesCheckbox = self.findChild(QCheckBox, "detectEllipsesCheckbox")
        self.detectCirclesCheckbox = self.findChild(QCheckBox, "detectCirclesCheckbox")
        self.detectLinesCheckbox = self.findChild(QCheckBox, "detectLinesCheckbox")

        self.detectEllipsesCheckbox.stateChanged.connect(self.detect_ellipses)
        self.detectCirclesCheckbox.stateChanged.connect(self.detect_circles)
        self.detectLinesCheckbox.stateChanged.connect(self.detect_lines)


        # Initialize Reset Button
        self.reset_button = self.findChild(QPushButton, "reset")
        self.reset_button.clicked.connect(self.reset_image)
        
        # Initialize Apply Snake Button
        self.apply_snake_greedy_button = self.findChild(QPushButton , "snakeApplyButton")
        self.apply_snake_greedy_button.clicked.connect(self.apply_snake_greedy)
        
        self.contour_perimeter = self.findChild(QLabel , "snakePerimeter")
        self.contour_area = self.findChild(QLabel , "snakeArea")
        self.chain_code = self.findChild(QLabel , "snakeCodeChain")
        
        # Initialize Snake Contour Drawing Shape
        self.drawing_mode_combobox = self.findChild(QComboBox , "snakeModelModeCombobox")
        self.drawing_mode_combobox.currentIndexChanged.connect(self.select_contour_drawing_mode)
        
        self.drawing_shapes_frame = self.findChild(QFrame , "predefinedShapesFrame")
        self.drawing_shapes_frame.hide()
        
        self.circle_predefined_shape_button = self.findChild(QPushButton , "circleButton")
        self.circle_predefined_shape_button.clicked.connect(self.select_circle_predefined_shape)
        
        self.rectangle_predefined_shape_button = self.findChild(QPushButton , "squareButton")
        self.rectangle_predefined_shape_button.clicked.connect(self.select_rectangle_predefined_shape)
        
        # Initialize Snake Parameters
        # Alpha
        self.snake_alpha = 1
        self.snake_alpha_line_edit = self.findChild(QLineEdit , "snakeAlphaInput")
        self.snake_alpha_line_edit.setText("1.0")
        self.snake_alpha_line_edit.textChanged.connect(self.edit_snake_alpha)
        
        # Beta
        self.snake_beta = 1
        self.snake_beta_line_edit = self.findChild(QLineEdit , "snakeBetaInput")
        self.snake_beta_line_edit.setText("1.0")
        self.snake_beta_line_edit.textChanged.connect(self.edit_snake_beta)
        
        # Gamma
        self.snake_gamma = 1
        self.snake_gamma_line_edit = self.findChild(QLineEdit , "snakeGammaInput")
        self.snake_gamma_line_edit.setText("1.0")
        self.snake_gamma_line_edit.textChanged.connect(self.edit_snake_gamma)
        
        # Window Size
        self.snake_window_size = 1
        self.snake_window_size_line_edit = self.findChild(QLineEdit , "snakeWindowSizeInput")
        self.snake_window_size_line_edit.setText("5")
        self.snake_window_size_line_edit.textChanged.connect(self.edit_snake_window_size)
        
        # Initialize Controller
        self.controller = Controller(self.input_image , self.output_image ,
                                    self.contour_drawing_widget , self.output_contour_drawing)
        
    def browse_image(self):
        self.controller.browse_input_image()
        self.controller.update_contour_drawing_widget_params()

    def reset_image(self):
        self.sigma_input.setText("1")
        self.t_low_input.setText("40")
        self.t_high_input.setText("100")

        # Update the output image display
        self.output_contour_drawing.pixmap = self.controller.numpy_to_qpixmap(self.input_image.input_image)
        self.output_contour_drawing.setPixmap(self.output_contour_drawing.pixmap)
        self.output_contour_drawing.setScaledContents(True)
        self.output_contour_drawing.image = self.output_contour_drawing.pixmap.copy()
        self.output_contour_drawing.contour_points.clear()
        self.output_contour_drawing.update()

    def apply_snake_greedy(self):
        self.controller.apply_snake_greedy(self.snake_alpha , self.snake_beta , self.snake_gamma , self.snake_window_size)
        self.update_contour_perimeter()
        self.update_contour_area()  
        self.update_chain_code()
        
    def update_contour_area(self):
        self.contour_area.setText(str(self.controller.snake.contour_area))
    
    def update_contour_perimeter(self):
        self.contour_perimeter.setText(str(self.controller.snake.contour_perimiter))
        
    def update_chain_code(self):
        self.chain_code.setText(str(self.controller.snake.chain_code))
        
    # def apply_canny_edge_detection(self):
        
    #     # Default values if fields are empty
    #     sigma = 1.4
    #     t_low = 40
    #     t_high = 100
        
    #     # Try to convert inputs to appropriate values
    #     try:
    #         if self.sigma_input.text():
    #             sigma = float(self.sigma_input.text())
    #         if self.t_low_input.text():
    #             t_low = float(self.t_low_input.text())
    #         if self.t_high_input.text():
    #             t_high = float(self.t_high_input.text())
    #     except ValueError:
    #         # If conversion fails, use default values
    #         pass
        
    #     # Apply Canny edge detection
    #     if self.input_image.input_image is not None:
    #         from classes.canny import apply_canny_edge_detection
            
    #         # Apply the Canny edge detection
    #         edges = apply_canny_edge_detection(self.input_image.input_image, sigma, t_low, t_high)
            
    #         # Convert the edges to 3-channel RGB for display
    #         if len(edges.shape) == 2:  # If single channel
    #             edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    #         else:
    #             edges_rgb = edges
            
    #         # Update the output image
    #         self.output_image.input_image = edges_rgb
    #         self.output_image.output_image = edges_rgb
            
    #         # Update the output image display
    #         self.output_contour_drawing.setPixmap(self.controller.numpy_to_qpixmap(edges_rgb))
    #         self.output_contour_drawing.setScaledContents(True)

    def apply_canny_edge_detection(self):
        
        sigma = float(self.sigma_input.text())
        low_threshold = float(self.t_low_input.text())
        high_threshold = float(self.t_high_input.text())

        detect_lines = self.isCannyLinesDetected
        detect_circles = self.isCannyCirclesDetected
        detect_ellipses = self.isCannyEllipsesDetected

        line_vote_threshold = self.cannyMaxVoteLinesValue
        circle_vote_threshold = self.cannyMaxVoteCirclesValue
        ellipse_vote_threshold = self.cannyMaxVoteEllipsesValue

        result_image = self.controller.apply_canny_edge_detection(sigma, low_threshold, high_threshold,
                              detect_lines, detect_circles, detect_ellipses,
                              line_vote_threshold, circle_vote_threshold, ellipse_vote_threshold)
        
        self.output_contour_drawing.pixmap = self.controller.numpy_to_qpixmap(result_image)
        self.output_contour_drawing.setPixmap(self.output_contour_drawing.pixmap)
        self.output_contour_drawing.setScaledContents(True)
        self.output_contour_drawing.image = self.output_contour_drawing.pixmap.copy()
        self.output_contour_drawing.update()

    def update_sigma(self):
            try:
                self.sigma_value = float(self.sigma_input.text())
            except ValueError:
                self.sigma_value = 1.0
                self.sigma_input.setText("1")
    
    def update_t_low(self):
        try:
            self.t_low_value = float(self.t_low_input.text())
        except ValueError:
            self.t_low_value = 40
            self.t_low_input.setText("40")

    def update_t_high(self):
        try:
            self.t_high_value = float(self.t_high_input.text())
        except ValueError:
            self.t_high_value = 100
            self.t_high_input.setText("100")

    def update_canny_max_vote_ellipses(self):
        self.cannyMaxVoteEllipses.setText(str(self.cannyMaxVoteEllipsesSlider.value()))
        self.cannyMaxVoteEllipsesValue = self.cannyMaxVoteEllipsesSlider.value()

    def update_canny_max_vote_circles(self):
        self.cannyMaxVoteCircles.setText(str(self.cannyMaxVoteCirclesSlider.value()))
        self.cannyMaxVoteCirclesValue = self.cannyMaxVoteCirclesSlider.value()

    def update_canny_max_vote_lines(self):
        self.cannyMaxVoteLines.setText(str(self.cannyMaxVoteLinesSlider.value()))
        self.cannyMaxVoteLinesValue = self.cannyMaxVoteLinesSlider.value()

    def detect_ellipses(self):
        self.isCannyEllipsesDetected = not self.isCannyEllipsesDetected

    def detect_circles(self):
        self.isCannyCirclesDetected = not self.isCannyCirclesDetected

    def detect_lines(self):
        self.isCannyLinesDetected = not self.isCannyLinesDetected

    def select_contour_drawing_mode(self , index):
        if(index == 1):
            self.contour_drawing_widget.current_mode = ContourMode.RECTANGLE
            self.drawing_shapes_frame.show()
        elif(index == 2):
            self.contour_drawing_widget.current_mode = ContourMode.FREE
            self.drawing_shapes_frame.hide()

    def select_circle_predefined_shape(self):
        self.contour_drawing_widget.current_mode = ContourMode.CIRCLE
    
    def select_rectangle_predefined_shape(self):
        self.contour_drawing_widget.current_mode = ContourMode.RECTANGLE
        
    def edit_snake_alpha(self , text):
        try:
            self.snake_alpha = float(text)
        except ValueError:
            self.snake_alpha = 1.0
            self.snake_alpha_line_edit.setText("1.0")
    
    def edit_snake_beta(self , text):
        try:
            self.snake_beta = float(text)
        except ValueError:
            self.snake_beta = 1.0
            self.snake_beta_line_edit.setText("1.0")
    
    def edit_snake_gamma(self , text):
        try:
            self.snake_gamma = float(text)
        except ValueError:
            self.snake_gamma = 1.0
            self.snake_gamma_line_edit.setText("1.0")
    
    def edit_snake_window_size(self , text):
        try:
            self.snake_window_size = int(text)
        except ValueError:
            self.snake_window_size = 5
            self.snake_window_size_line_edit.setText("5")
    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())   