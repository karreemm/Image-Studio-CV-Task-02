import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow , QPushButton , QStackedWidget ,QFrame , QLabel , QVBoxLayout, QHBoxLayout , QCheckBox , QComboBox , QLineEdit
from PyQt5.uic import loadUi
from PyQt5.QtGui import QIcon
from helper_functions.compile_qrc import compile_qrc
from classes.image import Image
from classes.controller import Controller
from classes.ImageEnum import ImageSource , Channel , DistributionCurve
from classes.statisticsVisualization import HistogramCanvas , CDFCanvas
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

compile_qrc()
from icons_setup.icons import *

from icons_setup.compiledIcons import *
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi('main.ui', self)
        self.setWindowTitle('Image Studio')
        self.setWindowIcon(QIcon('icons_setup\icons\logo.png'))

            
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())