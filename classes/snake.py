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

    def convert_list_to_qpoints(self, points_list):
        """
        Convert a list of (x, y) tuples back to a list of QPoint objects.
        """
        return [QPoint(x, y) for x, y in points_list]

    def external_energy(self, gradients, x, y):
        """Compute external energy (gradient magnitude) at (x, y)."""
        gradient_x, gradient_y = gradients
        result = np.square(np.abs(gradient_x[y, x])) + np.square(np.abs( gradient_y[y, x])) # Negative for attraction
        return -1 * result
        
    def internal_energy(self, contour, i, new_x, new_y):
        """Compute internal energy to maintain elasticity."""
        prev_point = contour[i - 1]
        next_point = contour[(i + 1) % len(contour)]  # Ensuring cyclic boundary condition

        elasticity = np.square(np.linalg.norm([next_point[0] - contour[i][0],next_point[1] - contour[i][1]]))

        curvature = np.square(np.linalg.norm([next_point[0] - 2 * contour[i][0] + prev_point[0], 
                                            next_point[1] - 2 * contour[i][1] + prev_point[1]]))
        
        return (elasticity , curvature)

    
    def active_contour_greedy(self , image, contour_points, alpha=4, beta=1, gamma=1, max_iterations=100, 
                          search_window_size=5):
        """
        Implement the active contour (snake) algorithm using a greedy approach.
        
        Parameters:
        -----------
        image : numpy.ndarray
            The input image as a 2D numpy array (grayscale).
        contour_points : list of tuples
            Initial contour points as a list of (x, y) tuples.
        alpha : float
            Weight for the continuity energy term.
        beta : float
            Weight for the curvature energy term.
        gamma : float
            Weight for the image (edge) energy term.
        max_iterations : int
            Maximum number of iterations for the snake evolution.
        search_window_size : int
            Size of the neighborhood to search for optimal points.
        
        Returns:
        --------
        numpy.ndarray
            Final snake contour points as a numpy array of shape (n, 2).
        """
        # Convert contour points to numpy array
        snake = np.array(contour_points)
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Precompute image gradient for edge energy
        gradient_x = np.zeros(image.shape, dtype=np.float64)
        gradient_y = np.zeros(image.shape, dtype=np.float64)
        
        # Simple x and y gradients using finite differences
        gradient_x[:, 1:-1] = image[:, 2:] - image[:, :-2]
        gradient_y[1:-1, :] = image[2:, :] - image[:-2, :]
        
        # Compute gradient magnitude
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Normalize gradient magnitude to [0, 1]
        if gradient_magnitude.max() > 0:
            gradient_magnitude = gradient_magnitude / gradient_magnitude.max()
        
        # Invert the gradient magnitude to use as image energy
        image_energy = 1.0 - gradient_magnitude
        
        # Main loop for snake evolution
        for iteration in range(max_iterations):
            new_snake = np.copy(snake)
            snake_energy = np.zeros(len(snake))
            
            # Loop through all points in the snake
            for i in range(len(snake)):
                # Get coordinates of the current point
                x, y = snake[i]
                
                # Get indices of previous and next points (circular)
                prev_idx = (i - 1) % len(snake)
                next_idx = (i + 1) % len(snake)
                
                # Previous and next points
                prev_point = snake[prev_idx]
                next_point = snake[next_idx]
                
                min_energy = float('inf')
                best_point = (x, y)
                
                # Search in the neighborhood
                for dx in range(-search_window_size, search_window_size + 1):
                    for dy in range(-search_window_size, search_window_size + 1):
                        new_x = int(x + dx)
                        new_y = int(y + dy)
                        
                        # Check if the new point is within image boundaries
                        if 0 <= new_x < width and 0 <= new_y < height:
                            # Calculate all energy terms
                            
                            # Continuity energy: distance to previous point
                            dist_prev = np.sqrt((new_x - prev_point[0])**2 + (new_y - prev_point[1])**2)
                            continuity_energy = dist_prev
                            
                            # Curvature energy: second derivative approximation
                            curvature_energy = ((prev_point[0] - 2*new_x + next_point[0])**2 + 
                                            (prev_point[1] - 2*new_y + next_point[1])**2)
                            
                            # Image energy: based on gradient magnitude
                            image_term = image_energy[new_y, new_x]
                            
                            # Total energy: weighted sum of the three energy terms
                            total_energy = (alpha * continuity_energy + 
                                        beta * curvature_energy + 
                                        gamma * image_term)
                            
                            # Update best point if we found a lower energy
                            if total_energy < min_energy:
                                min_energy = total_energy
                                best_point = (new_x, new_y)
                                snake_energy[i] = min_energy
                
                # Update snake point with the best one found
                new_snake[i] = best_point
            
            # Update the snake
            snake = new_snake
            
            # Check for convergence (when average energy change is small)
            if iteration > 0 and np.mean(snake_energy) < 0.01:
                print(f"Converged after {iteration+1} iterations")
                break
        
        return snake
    
    
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
