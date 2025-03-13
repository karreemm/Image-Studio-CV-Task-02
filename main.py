import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow , QPushButton , QStackedWidget ,QFrame , QLabel , QVBoxLayout, QHBoxLayout , QCheckBox , QComboBox , QLineEdit
from PyQt5.uic import loadUi
from PyQt5.QtGui import QIcon
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
        
        # Initialize Controller
        self.controller = Controller(self.input_image , self.output_image ,
                                    self.contour_drawing_widget , self.output_image_label)
        
    def browse_image(self):
        self.controller.browse_input_image()
        self.controller.update_contour_drawing_widget_params()
            
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())