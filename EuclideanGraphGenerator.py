import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class EuclideanGraphGenerator:
    def __init__(self, n=10, width=100, height=100, seed=None):
        self.n = n
        self.width = width
        self.height = height
        self.seed = seed

    def generate(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        
        # Generate random points
        points = np.random.rand(self.n, 2)
        points[:, 0] *= self.width
        points[:, 1] *= self.height
        
        # Calculate distance matrix
        dist_matrix = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    dist = np.linalg.norm(points[i] - points[j])
                    dist_matrix[i][j] = dist
        
        return dist_matrix, points

    def plot(self, points, tour=None, title="Euclidean Graph"):
        plt.figure(figsize=(8, 6))
        plt.scatter(points[:, 0], points[:, 1], c='blue', label='Nodes')
        
        for i, p in enumerate(points):
            plt.annotate(str(i), (p[0], p[1]), xytext=(5, 5), textcoords='offset points')
            
        if tour:
            tour_points = points[tour]
            # Add start point to end to close the loop for plotting
            tour_points = np.vstack([tour_points, tour_points[0]])
            plt.plot(tour_points[:, 0], tour_points[:, 1], 'r-', alpha=0.6, label='Tour')
            
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()
