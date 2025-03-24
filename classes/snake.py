from PyQt5.QtWidgets import QLabel, QVBoxLayout, QFrame, QPushButton, QHBoxLayout, QWidget ,QApplication ,QMessageBox
from PyQt5.QtCore import Qt, QPoint
import cv2
import numpy as np
from classes.canny import convolve
class Snake():
    def __init__(self):
        self.contour_points = []
        self.contour_perimiter = 0
        self.contour_area = 0
        self.chain_code = []
        
    def convert_qpoints_to_list(self, qpoints):
        
        return [(point.x(), point.y()) for point in qpoints]        

    def convert_list_to_qpoints(self, points_list):
        """
        Convert a list of (x, y) tuples back to a list of QPoint objects.
        """
        return [QPoint(x, y) for x, y in points_list]

    
    def compute_image_energy(self, image):
        """
        Compute the image gradient and normalize the gradient magnitude.
        """
        gradient_y , gradient_x = self.calculate_gradient_sobel(image)
        external_energy_magnitude = np.sqrt((gradient_x)**2 +(gradient_y)**2)
        return -external_energy_magnitude**2
    
    def compute_weighted_internal_energy(self, previous_point, new_x, new_y, next_point, alpha, beta):
        """
        Compute the internal energy (continuity and curvature).
        """
        elasticity_energy = ((new_x - previous_point[0])**2 + (new_y - previous_point[1])**2 + (next_point[0] - new_x)**2 + (next_point[1] - new_y)**2)
        curvature_energy = ((previous_point[0] - 2 * new_x + next_point[0])**2 +
                            (previous_point[1] - 2 * new_y + next_point[1])**2)
        return alpha * elasticity_energy + beta * curvature_energy

    def compute_weighted_external_energy(self, image_energy, new_x, new_y, gamma):
        """
        Compute the external energy based on the image gradient.
        """
        return gamma * image_energy[new_y, new_x]

    
    def active_contour_greedy(self, image, output_image_label ,alpha=1, beta=1, gamma=1, max_iterations=1000, 
                            search_window_size=5):
        """
        Implement the active contour (snake) algorithm using a greedy approach.
        """
        height, width = image.shape[:2]
        image_energy = self.compute_image_energy(image)
        if search_window_size % 2 == 0:
            search_window_size += 1
        search_window_size = (search_window_size - 1) // 2
        
        for iteration in range(max_iterations): 
            moved = False           
            new_snake = list(self.contour_points)
            for i in range(len(self.contour_points)):
                x, y = self.contour_points[i]
                previous_index, next_index = (i - 1) % len(self.contour_points), (i + 1) % len(self.contour_points)
                previous_point, next_point = self.contour_points[previous_index], self.contour_points[next_index]
                
                min_energy, optimal_point = float('inf'), (x, y)
                
                for dx in range(-search_window_size, search_window_size + 1):
                    for dy in range(-search_window_size, search_window_size + 1):
                        new_x, new_y = int(x + dx), int(y + dy)
                        
                        if 0 <= new_x < width and 0 <= new_y < height:
                            internal_energy = self.compute_weighted_internal_energy(previous_point, new_x, new_y, next_point, alpha, beta)
                            external_energy = self.compute_weighted_external_energy(image_energy, new_x, new_y, gamma)
                            total_energy = internal_energy + external_energy
                            
                            if total_energy < min_energy:
                                min_energy, optimal_point = total_energy, (new_x, new_y)
                
                if(optimal_point != (x,y)):
                    new_snake[i] = optimal_point
                    moved = True
            
            self.contour_points = new_snake
            new_contour_points = self.convert_list_to_qpoints(new_snake)
            output_image_label.contour_points = new_contour_points
            output_image_label.update()
            self.resample_contour_points()
            QApplication.processEvents()
            if not moved:
                break
            
            # if iteration > 0 and np.mean(snake_energy) < 0.0001:
            #     print(f"Converged after {iteration + 1} iterations")
            #     break
        QMessageBox.information(None, "Iterations Ended", "Iterations Ended")
        self.contour_perimiter = round(self.calculate_contour_perimiter(new_snake))
        self.contour_area = round(self.calculate_contour_area(new_snake))
        self.generate_chain_code(new_snake)
        


    def calculate_contour_perimiter(self, contour):
        """Calculate the perimiter of the contour."""
        contour_perimeter = 0.0
        for i in range(len(contour)):
            p1 = np.array(contour[i])
            p2 = np.array(contour[(i + 1) % len(contour)])  # Wrap around for closed contour
            contour_perimeter += np.linalg.norm(p1 - p2)
        return contour_perimeter
    
    
    def calculate_contour_area(self, contour):
        """Calculate the contour_area enclosed by the contour."""
        contour_area = 0
        for i in range(len(contour)):
            contour_area += contour[i][0] * contour[(i + 1) % len(contour)][1] - contour[i][1] * contour[(i + 1) % len(contour)][0]
        return 0.5 * np.abs(contour_area)
    
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
        formatted_chain_code = " ".join(self.chain_code[i:i+6] for i in range(0, len(self.chain_code), 6))
        self.chain_code = formatted_chain_code
        
    def calculate_gradient_sobel(self , image):
        vertical_edges = cv2.Sobel(image , cv2.CV_64F , 0, 1,ksize=3)
        horizontal_edges = cv2.Sobel(image , cv2.CV_64F , 1, 0, ksize=3)

        return vertical_edges , horizontal_edges
    
    def apply_gaussian_blur(self , image , filter_size , sigma = 1):
        x, y = np.meshgrid(np.arange(-filter_size // 2,(filter_size // 2 )+1), np.arange(-filter_size // 2,(filter_size // 2 )+1))  
        kernel = np.exp(-(x**2 + y**2)/(2*sigma**2))/(2*np.pi*sigma**2) 
        kernel = kernel / np.sum(kernel)
        return convolve(image , kernel)
    
    def resample_contour_points(self):
        """
        Resample the given contour points to be evenly spaced.
        """
        if len(self.contour_points) < 2:
            return self.contour_points

        x_vals = np.array([p[0] for p in self.contour_points])
        y_vals = np.array([p[1] for p in self.contour_points])

        # Compute cumulative distances
        distances = np.sqrt(np.diff(x_vals) ** 2 + np.diff(y_vals) ** 2)
        cumulative_distances = np.insert(np.cumsum(distances), 0, 0)

        # Generate evenly spaced distances
        new_distances = np.linspace(0, cumulative_distances[-1], len(self.contour_points))

        # Interpolate new points
        new_x_vals = np.interp(new_distances, cumulative_distances, x_vals)
        new_y_vals = np.interp(new_distances, cumulative_distances, y_vals)

        return [(int(x), int(y)) for x, y in zip(new_x_vals, new_y_vals)]
        
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
