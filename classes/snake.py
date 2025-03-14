from PyQt5.QtWidgets import QLabel, QVBoxLayout, QFrame, QPushButton, QHBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, QPoint
import cv2
import numpy as np
from classes.enums import ContourMode

class Snake():
    def __init__(self):
        self.contour_points = []  # Free-draw contour points
    
    
    def convert_qpoints_to_list(self, qpoints):
        
        self.contour_points = [(point.x(), point.y()) for point in qpoints]        


    def external_energy(self, image, x, y):
        """Compute external energy (gradient magnitude) at (x, y)."""
        return -cv2.magnitude(image[y, x, 0], image[y, x, 1])  # Negative for attraction

    def internal_energy(self, contour, i, new_x, new_y):
        """Compute internal energy to maintain smoothness."""
        prev_point = contour[i - 1]
        next_point = contour[(i + 1) % len(contour)]
        
        continuity = np.linalg.norm([new_x - prev_point[0], new_y - prev_point[1]])
        curvature = np.linalg.norm([new_x - 2 * contour[i][0] + next_point[0], 
                                    new_y - 2 * contour[i][1] + next_point[1]])
        
        return continuity + curvature

    def greedy_snake(self, image, contour, alpha=0.1, beta=0.1, iterations=100):
        """Greedy algorithm to adjust contour points within a 5x5 window."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        for _ in range(iterations):
            new_contour = []
            for i, (x, y) in enumerate(contour):
                min_energy = float('inf')
                best_x, best_y = x, y
                
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        new_x, new_y = x + dx, y + dy
                        
                        if 0 <= new_x < image.shape[1] and 0 <= new_y < image.shape[0]:
                            E_ext = self.external_energy((gradient_x, gradient_y), new_x, new_y)
                            E_int = self.internal_energy(contour, i, new_x, new_y)
                            total_energy = alpha * E_int + beta * E_ext
                            
                            if total_energy < min_energy:
                                min_energy = total_energy
                                best_x, best_y = new_x, new_y
                
                new_contour.append((best_x, best_y))
            
            contour = np.array(new_contour, dtype=np.int32)

        return contour
