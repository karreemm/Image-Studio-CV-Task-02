from PyQt5.QtWidgets import QLabel, QVBoxLayout, QFrame, QPushButton, QHBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, QPoint
import cv2
import numpy as np
from classes.enums import ContourMode

class ContourDrawingWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.image = None  # Original image
        self.pixmap = None  # Pixmap for drawing
        self.drawing = False  # Flag to check if drawing is active
        self.start_point = None  # Start point of shape
        self.end_point = None  # End point of shape
        self.contour_points = []  # Free-draw contour points
        self.current_mode = ContourMode.RECTANGLE  # Default mode (free draw)


    # def set_mode(self, mode):
    #     """Change contour mode (free, rectangle, circle)."""
    #     self.current_mode = mode
    #     self.contour_points = []  # Reset points
    #     self.start_point = None
    #     self.end_point = None
    #     self.update()

    def mousePressEvent(self, event):
        """Handles mouse press event to start drawing."""
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.start_point = event.pos()
            if self.current_mode == ContourMode.FREE:
                self.contour_points = [event.pos()]  # Start free-draw

    def mouseMoveEvent(self, event):
        """Handles mouse movement for dynamic drawing."""
        if self.drawing:
            if self.current_mode == ContourMode.FREE:
                self.contour_points.append(event.pos())  # Add points dynamically
            else:
                self.end_point = event.pos()  # Update shape end point
            self.update()

    def mouseReleaseEvent(self, event):
        """Stops drawing when the mouse is released."""
        if event.button() == Qt.LeftButton:
            self.drawing = False
            self.update()

    def paintEvent(self, event):
        """Redraws the image and overlays contours."""
        if self.pixmap:
            painter = QPainter(self)
            
            # scale_w = self.width() / self.pixmap.width()
            # scale_h = self.height() / self.pixmap.height()
        
            painter.drawPixmap(self.rect(), self.pixmap)

            pen = QPen(Qt.red, 2, Qt.SolidLine)
            painter.setPen(pen)

            if self.current_mode == ContourMode.FREE and self.contour_points:
                for i in range(len(self.contour_points) - 1):
                    p1 = self.contour_points[i]
                    p2 = self.contour_points[i + 1]
                    painter.drawLine(p1, p2)

            elif self.current_mode == ContourMode.RECTANGLE and self.start_point and self.end_point:
                rect_x = min(self.start_point.x(), self.end_point.x())
                rect_y = min(self.start_point.y(), self.end_point.y())
                rect_width = abs(self.start_point.x() - self.end_point.x())
                rect_height = abs(self.start_point.y() - self.end_point.y())
                painter.drawRect(rect_x, rect_y, rect_width, rect_height)

            elif self.current_mode == ContourMode.CIRCLE and self.start_point and self.end_point:
                center_x = (self.start_point.x() + self.end_point.x()) // 2
                center_y = (self.start_point.y() + self.end_point.y()) // 2
                radius = int(((self.start_point.x() - self.end_point.x()) ** 2 +
                              (self.start_point.y() - self.end_point.y()) ** 2) ** 0.5) // 2
                painter.drawEllipse(QPoint(center_x, center_y), radius, radius)

            painter.end()
