import numpy as np
import random
import networkx as nx

class NonMetricGraphGenerator:
    """
    Generates random non-Euclidean graphs.
    Tuned to produce graphs where:
    1. Generalized TSP (Walk) < Standard TSP (Cycle)
    2. Standard TSP <= 2 * Generalized TSP
    3. The three costs (TSP, Walk, TSP-Walk) are often distinct.
    """
    def __init__(self, n=8, num_clusters=3, low_cost_range=(10, 30), high_cost_range=(40, 80), seed=None):
        self.n = n
        self.num_clusters = max(2, num_clusters)
        self.low_cost_range = low_cost_range
        self.high_cost_range = high_cost_range
        self.seed = seed

    def generate(self):
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)

        # 1. Initialize with High Costs
        dist_matrix = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i + 1, self.n):
                cost = int(random.uniform(*self.high_cost_range))
                dist_matrix[i][j] = cost
                dist_matrix[j][i] = cost

        # 2. Assign nodes to clusters
        nodes = list(range(self.n))
        random.shuffle(nodes)
        clusters = np.array_split(nodes, self.num_clusters)
        
        # 3. Intra-cluster connections (Low Cost)
        for cluster in clusters:
            if len(cluster) < 1: continue
            for i in range(len(cluster)):
                for j in range(i + 1, len(cluster)):
                    u, v = cluster[i], cluster[j]
                    cost = int(random.uniform(*self.low_cost_range))
                    dist_matrix[u][v] = cost
                    dist_matrix[v][u] = cost

        # 4. Chain connections (Low Cost Bridges)
        # Connect C[i] to C[i+1]
        for k in range(self.num_clusters - 1):
            c1 = clusters[k]
            c2 = clusters[k+1]
            
            # Create a few bridges to allow flexibility
            num_bridges = min(len(c1) * len(c2), 2)
            for _ in range(num_bridges):
                u = random.choice(c1)
                v = random.choice(c2)
                cost = int(random.uniform(*self.low_cost_range))
                dist_matrix[u][v] = cost
                dist_matrix[v][u] = cost

        # 5. Add some random "shortcuts" or noise to make it less generic
        # This helps in making TSP perm != Walk perm
        num_noise = self.n
        for _ in range(num_noise):
            u, v = random.sample(range(self.n), 2)
            if dist_matrix[u][v] > self.high_cost_range[0]: # If it"s a high cost edge
                # Make it medium cost
                cost = int(random.uniform(self.low_cost_range[1], self.high_cost_range[0]))
                dist_matrix[u][v] = cost
                dist_matrix[v][u] = cost

        # 5. Ensure diagonal is 0
        np.fill_diagonal(dist_matrix, 0)
        
        return dist_matrix
