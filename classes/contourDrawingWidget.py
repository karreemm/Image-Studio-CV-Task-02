import math
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtCore import Qt, QPoint
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
        self.current_mode = ContourMode.FREE  # Default mode (free draw)
        self.drawing_enabled = True
    
    def setDrawingEnabled(self, enabled):
        """Enable or disable drawing functionality."""
        self.drawing_enabled = enabled

    def mousePressEvent(self, event):
        """Handles mouse press event to start drawing."""
        if (self.drawing_enabled == False):
            return
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.start_point = event.pos()
            
            if self.current_mode == ContourMode.FREE:
                self.contour_points = [event.pos()]  # Start free-draw
            
            elif self.current_mode in (ContourMode.RECTANGLE, ContourMode.CIRCLE):
                self.contour_points = []  # Reset contour points

    def mouseMoveEvent(self, event):
        """Handles mouse movement for dynamic drawing."""
        if (self.drawing_enabled == False):
            return
        if self.drawing:
            if self.current_mode == ContourMode.FREE:
                self.contour_points.append(event.pos())  # Add points dynamically
            else:
                self.end_point = event.pos()  # Update shape end point
                
                if self.current_mode == ContourMode.RECTANGLE:
                    self.update_rectangle_contour()

                elif self.current_mode == ContourMode.CIRCLE:
                    self.update_circle_contour()
            
            self.update()
            
    def mouseReleaseEvent(self, event):
        """Stops drawing when the mouse is released."""
        if (self.drawing_enabled == False):
            return
        if event.button() == Qt.LeftButton:
            self.drawing = False
            self.update()

    def paintEvent(self, event):
        """Redraws the image and overlays contours using stored contour points."""
        if self.pixmap:
            painter = QPainter(self)
            painter.drawPixmap(self.rect(), self.pixmap)

            pen = QPen(Qt.red, 2, Qt.SolidLine)
            painter.setPen(pen)

            # Draw Freehand Contour
            if self.current_mode == ContourMode.FREE and len(self.contour_points) > 1:
                for i in range(len(self.contour_points) - 1):
                    p1 = self.contour_points[i]
                    p2 = self.contour_points[i + 1]
                    painter.drawLine(p1, p2)
                # painter.drawLine(self.contour_points[-1], self.contour_points[0])

            # Draw Rectangle Using Contour Points
            elif self.current_mode == ContourMode.RECTANGLE and len(self.contour_points) > 1:
                for i in range(len(self.contour_points) - 1):
                    p1 = self.contour_points[i]
                    p2 = self.contour_points[i + 1]
                    painter.drawLine(p1, p2)
                # Close the rectangle by connecting the last and first point
                painter.drawLine(self.contour_points[-1], self.contour_points[0])

            # Draw Circle Using Contour Points
            elif self.current_mode == ContourMode.CIRCLE and len(self.contour_points) > 1:
                for point in self.contour_points:
                    painter.drawPoint(point)  # Draw small points forming the circular contour
                for i in range(len(self.contour_points) - 1):
                    p1 = self.contour_points[i]
                    p2 = self.contour_points[i + 1]
                    painter.drawLine(p1, p2)
                # Connect last and first points to complete the shape
                painter.drawLine(self.contour_points[-1], self.contour_points[0])
                
            painter.end()


    def update_rectangle_contour(self):
        """Updates contour points for a rectangle with intermediate points."""
        x1, y1 = self.start_point.x(), self.start_point.y()
        x2, y2 = self.end_point.x(), self.end_point.y()

        num_points_per_edge = 100  # More points for smoother edges

        # Function to generate intermediate points between two points
        def interpolate_points(p1, p2, num_points):
            return [QPoint(int(p1.x() + (p2.x() - p1.x()) * t),
                        int(p1.y() + (p2.y() - p1.y()) * t))
                    for t in np.linspace(0, 1, num_points, endpoint=False)]

        # Define four corner points
        top_left = QPoint(x1, y1)
        top_right = QPoint(x2, y1)
        bottom_right = QPoint(x2, y2)
        bottom_left = QPoint(x1, y2)

        # Generate intermediate points along the edges
        top_edge = interpolate_points(top_left, top_right, num_points_per_edge)
        right_edge = interpolate_points(top_right, bottom_right, num_points_per_edge)
        bottom_edge = interpolate_points(bottom_right, bottom_left, num_points_per_edge)
        left_edge = interpolate_points(bottom_left, top_left, num_points_per_edge)

        # Store all points in contour
        self.contour_points = top_edge + right_edge + bottom_edge + left_edge
        
    def update_circle_contour(self):
        """Approximates a circle with multiple points."""
        center_x = (self.start_point.x() + self.end_point.x()) // 2
        center_y = (self.start_point.y() + self.end_point.y()) // 2
        radius = abs(self.start_point.x() - self.end_point.x()) // 2  # Approximate radius

        num_points = 600  # Increase for a smoother circle
        self.contour_points = [
            QPoint(int(center_x + radius * math.cos(theta)), 
                int(center_y + radius * math.sin(theta))) 
            for theta in np.linspace(0, 2 * math.pi, num_points)
        ]