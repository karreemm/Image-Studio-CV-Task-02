from PyQt5.QtWidgets import QLabel, QVBoxLayout, QFrame, QPushButton, QHBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, QPoint
import cv2
import numpy as np
from classes.enums import ContourMode

class Snake():
    def __init__(self):
        self.contour_points = []  # Free-draw contour points
        self.contour_perimiter = 0
        self.contour_area = 0
        self.chain_code = []
        
    def convert_qpoints_to_list(self, qpoints):
        
        self.contour_points = [(point.x(), point.y()) for point in qpoints]        

    def convert_list_to_qpoints(self, points_list):
        """
        Convert a list of (x, y) tuples back to a list of QPoint objects.
        """
        return [QPoint(x, y) for x, y in points_list]

    
    def compute_image_gradient(self, image):
        """
        Compute the image gradient and normalize the gradient magnitude.
        """
        gradient_x = np.zeros(image.shape, dtype=np.float64)
        gradient_y = np.zeros(image.shape, dtype=np.float64)
        gradient_x[:, 1:-1] = image[:, 2:] - image[:, :-2]
        gradient_y[1:-1, :] = image[2:, :] - image[:-2, :]
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        if gradient_magnitude.max() > 0:
            gradient_magnitude = gradient_magnitude / gradient_magnitude.max()
        
        # Invert the gradient magnitude to use as image energy
        return (1.0 - gradient_magnitude)
    
    def compute_internal_energy(self, previous_point, new_x, new_y, next_point, alpha, beta):
        """
        Compute the internal energy (continuity and curvature).
        """
        continuity_energy = np.sqrt((new_x - previous_point[0])**2 + (new_y - previous_point[1])**2)
        curvature_energy = ((previous_point[0] - 2 * new_x + next_point[0])**2 +
                            (previous_point[1] - 2 * new_y + next_point[1])**2)
        return alpha * continuity_energy + beta * curvature_energy

    def compute_external_energy(self, image_energy, new_x, new_y, gamma):
        """
        Compute the external energy based on the image gradient.
        """
        return gamma * image_energy[new_y, new_x]

    
    def active_contour_greedy(self, image, contour_points, alpha=4, beta=1, gamma=1, max_iterations=100, 
                            search_window_size=5):
        """
        Implement the active contour (snake) algorithm using a greedy approach.
        """
        snake = np.array(contour_points)
        height, width = image.shape[:2]
        image_energy = self.compute_image_gradient(image)
        
        for iteration in range(max_iterations):
            new_snake = np.copy(snake)
            snake_energy = np.zeros(len(snake))
            
            for i in range(len(snake)):
                x, y = snake[i]
                previous_index, next_index = (i - 1) % len(snake), (i + 1) % len(snake)
                previous_point, next_point = snake[previous_index], snake[next_index]
                
                min_energy, optimal_point = float('inf'), (x, y)
                
                for dx in range(-search_window_size, search_window_size + 1):
                    for dy in range(-search_window_size, search_window_size + 1):
                        new_x, new_y = int(x + dx), int(y + dy)
                        
                        if 0 <= new_x < width and 0 <= new_y < height:
                            internal_energy = self.compute_internal_energy(previous_point, new_x, new_y, next_point, alpha, beta)
                            external_energy = self.compute_external_energy(image_energy, new_x, new_y, gamma)
                            total_energy = internal_energy + external_energy
                            
                            if total_energy < min_energy:
                                min_energy, optimal_point = total_energy, (new_x, new_y)
                                snake_energy[i] = min_energy
                
                new_snake[i] = optimal_point
            
            snake = new_snake
            
            if iteration > 0 and np.mean(snake_energy) < 0.01:
                print(f"Converged after {iteration + 1} iterations")
                break
        
        self.contour_perimiter = round(self.calculate_contour_perimiter(snake))
        self.contour_area = round(self.calculate_contour_area(snake))
        print(f'contour_perimiter : {self.contour_perimiter}')
        print(f'contour_area : {self.contour_area}')
        
        return snake

    def calculate_contour_perimiter(self, contour):
        """Calculate the perimiter of the contour."""
        contour_perimiter = 0
        for i in range(len(contour)):
            contour_perimiter += np.linalg.norm(contour[i] - contour[(i + 1) % len(contour)])
        return contour_perimiter
    
    
    def calculate_contour_area(self, contour):
        """Calculate the contour_area enclosed by the contour."""
        contour_area = 0
        for i in range(len(contour)):
            contour_area += contour[i][0] * contour[(i + 1) % len(contour)][1] - contour[i][1] * contour[(i + 1) % len(contour)][0]
        return 0.5 * np.abs(contour_area)
    
    def generate_chain_code(self, contour):
        """Generate the chaincode for the contour."""
        chaincode = []
        for i in range(len(contour)):
            x_diff = contour[(i + 1) % len(contour)][0] - contour[i][0]
            y_diff = contour[(i + 1) % len(contour)][1] - contour[i][1]
            chaincode.append((x_diff, y_diff))
        return chaincode
    
    # def greedy_snake(self, image, contour, alpha=4, beta=1,gamma = 1, iterations=100):
    #     """Greedy algorithm to adjust contour points within a 5x5 window."""
        
    #     gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    #     gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
        
    #     for _ in range(iterations):
    #         new_contour = []
    #         for i, (x, y) in enumerate(contour):
    #             min_energy = float('inf')
    #             best_x, best_y = x, y
                
    #             for dx in range(-2, 3):
    #                 for dy in range(-2, 3):
    #                     new_x, new_y = x + dx, y + dy
                        
    #                     if 0 <= new_x < image.shape[1] and 0 <= new_y < image.shape[0]:
    #                         E_ext = self.external_energy((gradient_x, gradient_y), new_x, new_y)
    #                         elasticity , curvature  = self.internal_energy(contour, i, new_x, new_y)
    #                         total_energy =  alpha * elasticity +beta * curvature + gamma * E_ext
    #                         if total_energy < min_energy:
    #                             min_energy = total_energy
    #                             best_x, best_y = new_x, new_y
                
    #             new_contour.append((best_x, best_y))
            
    #         contour = np.array(new_contour, dtype=np.int32)

    #     return contour
