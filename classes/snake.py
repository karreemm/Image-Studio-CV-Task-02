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
        self.chain_code = ""
        
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
        
        self.calculate_contour_perimiter(snake)
        self.calculate_contour_area(snake)
        self.generate_chain_code(snake)
        self.show_difference(snake)
        return snake

    def calculate_contour_perimiter(self, contour):
        """Calculate the perimiter of the contour."""
        contour_perimiter = 0
        for i in range(len(contour)):
            contour_perimiter += np.linalg.norm(contour[i] - contour[(i + 1) % len(contour)])
        self.contour_perimiter = round(contour_perimiter)
    
    def calculate_contour_area(self, contour):
        """Calculate the contour_area enclosed by the contour."""
        contour_area = 0
        # for i in range(len(contour)):
        #     x1, y1 = contour[i]
        #     x2, y2 = contour[(i + 1) % len(contour)]
            
        #     # Check if points are within image bounds
        #     if x1 < 0 or x1 >= 256 or y1 < 0 or y1 >= 256:
        #         print(f"Point ({x1}, {y1}) is out of image bounds")
        #     if x2 < 0 or x2 >= 256 or y2 < 0 or y2 >= 256:
        #         print(f"Point ({x2}, {y2}) is out of image bounds")
            
        #     contour_area += x1 * y2 - y1 * x2
        
        self.contour_area = round(0.5 * np.abs(contour_area))
        print(f"Calculated contour area: {self.contour_area}")
    # def calculate_contour_area(self, contour):
    #     """Calculate the contour_area enclosed by the contour."""
    #     contour_area = 0
    #     for i in range(len(contour)):
    #         contour_area += contour[i][0] * contour[(i + 1) % len(contour)][1] - contour[i][1] * contour[(i + 1) % len(contour)][0]
    #     self.contour_area = round(0.5 * np.abs(contour_area))
    
    def generate_chain_code(self, contour):
        """Generate the chaincode for the contour."""
        self.chain_code = ""
        for i in range(len(contour)):
            
            x_diff = contour[(i + 1) % len(contour)][0] - contour[i][0]
            y_diff = contour[(i + 1) % len(contour)][1] - contour[i][1]
            
            if x_diff == 0 and y_diff > 0:     # Right
                for i in range(y_diff):
                    self.chain_code += '0'
            elif x_diff == 1 and y_diff == 1:   # Up right diagonal
                self.chain_code += '1'
            elif x_diff < 0 and y_diff == 0:   # Up
                for i in range(abs(x_diff)):
                    self.chain_code += '2'
            elif x_diff == 1 and y_diff == -1:  # Up left diagonal
                self.chain_code += '3'
            elif x_diff == 0 and y_diff < 0:  # Left
                for i in range(abs(y_diff)):
                    self.chain_code += '4'
            elif x_diff == -1 and y_diff == -1: # Down left diagonal
                self.chain_code += '5'
            elif x_diff > 0 and y_diff == 0:  # Down
                for i in range(x_diff):
                    self.chain_code += '6'
            elif x_diff == -1 and y_diff == 1:  # Down right diagonal
                self.chain_code += '7'
                
            elif x_diff > 1 and y_diff > 1:        # Down diagonal movement with right or bottom
                for i in range(min(x_diff, y_diff)):
                    self.chain_code += '7'
                if x_diff > y_diff:                 # Down
                    for i in range(x_diff - y_diff):
                        self.chain_code += '6'
                elif x_diff < y_diff:               # Right
                    for i in range(y_diff - x_diff):
                        self.chain_code += '0'
                        
            elif x_diff > 1 and y_diff < -1:       # Down diagonal movement with left or bottom
                for i in range(min(x_diff, abs(y_diff))):
                    self.chain_code += '5'
                if x_diff > abs(y_diff):            # Down
                    for i in range(x_diff - abs(y_diff)):
                        self.chain_code += '6'      
                elif x_diff < abs(y_diff):           # Left
                    for i in range(abs(y_diff) - x_diff):
                        self.chain_code += '4'
                        
            elif x_diff < -1 and y_diff < -1:      # Up Diagonal movement with left or top
                for i in range(min(abs(x_diff), abs(y_diff))):
                    self.chain_code += '3'
                if abs(x_diff) > abs(y_diff):                  # Up
                    for i in range(abs(x_diff) - abs(y_diff)):
                        self.chain_code += '2'
                elif abs(x_diff) < abs(y_diff):                # Left
                    for i in range(abs(y_diff) - abs(x_diff)): 
                        self.chain_code += '4'
                        
            elif x_diff < -1 and y_diff > 1:        # Up Diagonal movement with right or top
                for i in range(min(abs(x_diff), y_diff)):
                    self.chain_code += '1'
                if abs(x_diff) > y_diff:                  # Up
                    for i in range(abs(x_diff) - y_diff):
                        self.chain_code += '2'
                elif abs(x_diff) > y_diff:                # Right
                    for i in range(y_diff - abs(x_diff)):
                        self.chain_code += '0'
        # print(f'chain_code : {self.chain_code}')
        
    def show_difference(self, contour):
        """Show the difference between the image and the contour."""
        diff_in_x = []
        # diff_in_y = []
        # diff_in_x_y = []
        for i in range(len(contour)):
            
            x_diff = contour[(i + 1) % len(contour)][0] - contour[i][0]
            # y_diff = contour[(i + 1) % len(contour)][1] - contour[i][1]
            # if x_diff > 1 and y_diff > 1:
            #     diff_in_x_y.append((x_diff, y_diff))
            # elif x_diff > 3:
            diff_in_x.append(x_diff)
            # elif y_diff >3:
            #     diff_in_y.append(y_diff)
        # print(f'diff_in_x_y : {diff_in_x_y}')    
        print(f'diff_in_x : {diff_in_x}')
        # print(f'diff_in_y : {diff_in_y}')
                
                
        
    
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
