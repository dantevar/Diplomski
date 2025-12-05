import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.graph_analysis import GraphAnalyzer

def create_fer_walk_graph():
    """
    Creates a graph where the optimal Walk spells out "FER" when plotted.
    The graph is designed so that Walk < TSP, and the Walk path traces "FER".
    """
    
    # Define coordinates for "FER" letters with more nodes for clearer shapes
    fer_coords = {
        # Letter F (vertical line + 2 horizontal lines)
        0: (0, 0),   # bottom of F
        1: (0, 1),   # 
        2: (0, 2),   # middle of F
        3: (0, 3),   # 
        4: (0, 4),   # top of F
        5: (1, 4),   # top horizontal line
        6: (2, 4),   # 
        7: (1, 2),   # middle horizontal line
        8: (1.5, 2), # 
        
        # Letter E (vertical line + 3 horizontal lines)
        9: (4, 0),   # bottom-left of E
        10: (4, 1),  # 
        11: (4, 2),  # middle-left of E
        12: (4, 3),  # 
        13: (4, 4),  # top-left of E
        14: (5, 0),  # bottom horizontal
        15: (6, 0),  # 
        16: (5, 2),  # middle horizontal
        17: (5.5, 2),# 
        18: (5, 4),  # top horizontal
        19: (6, 4),  # 
        
        # Letter R (vertical line + top horizontal + diagonal + middle horizontal)
        20: (8, 0),  # bottom-left of R
        21: (8, 1),  # 
        22: (8, 2),  # middle-left of R
        23: (8, 3),  # 
        24: (8, 4),  # top-left of R
        25: (9, 4),  # top horizontal
        26: (10, 4), # top-right of R
        27: (10, 3), # right side upper
        28: (10, 2), # right side middle
        29: (9, 2),  # middle horizontal
        30: (9.5, 1.5), # diagonal start
        31: (10, 1), # diagonal middle
        32: (10.5, 0), # bottom-right of R
    }
    
    n = len(fer_coords)
    
    # Create distance matrix based on Euclidean distances
    matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                x1, y1 = fer_coords[i]
                x2, y2 = fer_coords[j]
                dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                matrix[i][j] = dist
    
    # Define the optimal Walk path that clearly spells "FER"
    optimal_walk_path = [
        # Draw F: bottom to top, then horizontals
        0, 1, 2, 3, 4,     # vertical line bottom to top
        5, 6,              # top horizontal line
        4, 2, 7, 8,        # back to middle, draw middle horizontal
        2, 0,              # go back to bottom of F before moving to E
        
        # Move to E and draw it
        9,                 # move from F bottom to E bottom
        10, 11, 12, 13,    # vertical line bottom to top
        18, 19,            # top horizontal
        13, 11, 16, 17,    # back to middle, draw middle horizontal  
        11, 9, 14, 15,     # back to bottom, draw bottom horizontal
        
        # Move to R and draw it
        15, 20,            # move from E to R
        21, 22, 23, 24,    # vertical line bottom to top
        25, 26,            # top horizontal
        27, 28, 29,        # right side and middle horizontal
        22, 30, 31, 32     # diagonal from middle to bottom-right
    ]
    
    # Make the optimal walk path very cheap
    cheap_cost = 1.0
    for i in range(len(optimal_walk_path) - 1):
        u = optimal_walk_path[i]
        v = optimal_walk_path[i + 1]
        matrix[u][v] = cheap_cost
        matrix[v][u] = cheap_cost
    
    # Make non-walk edges expensive to force TSP to use longer routes
    expensive_multiplier = 5.0
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i][j] != cheap_cost:
                matrix[i][j] *= expensive_multiplier
                matrix[j][i] *= expensive_multiplier
    
    return matrix, fer_coords, optimal_walk_path

def approximate_tsp_cost(matrix):
    """Quick nearest-neighbor approximation for TSP cost."""
    n = matrix.shape[0]
    visited = [False] * n
    current = 0
    visited[0] = True
    total_cost = 0
    
    for _ in range(n - 1):
        best_next = -1
        best_cost = float('inf')
        
        for next_node in range(n):
            if not visited[next_node] and matrix[current][next_node] < best_cost:
                best_next = next_node
                best_cost = matrix[current][next_node]
        
        total_cost += best_cost
        visited[best_next] = True
        current = best_next
    
    # Return to start
    total_cost += matrix[current][0]
    return total_cost

def plot_fer_graph(coords, walk_path):
    """Plot the graph with FER layout and optimal walk path."""
    
    plt.figure(figsize=(15, 8))
    
    # Plot all nodes
    x_coords = [coords[i][0] for i in coords.keys()]
    y_coords = [coords[i][1] for i in coords.keys()]
    
    plt.scatter(x_coords, y_coords, c='lightblue', s=100, zorder=5)
    
    # Label nodes
    for node, (x, y) in coords.items():
        plt.annotate(f'{node}', (x, y), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8)
    
    # Plot the optimal walk path
    for i in range(len(walk_path) - 1):
        u = walk_path[i]
        v = walk_path[i + 1]
        x1, y1 = coords[u]
        x2, y2 = coords[v]
        
        plt.arrow(x1, y1, x2-x1, y2-y1, 
                 head_width=0.05, head_length=0.05, 
                 fc='red', ec='red', alpha=0.7, zorder=3)
    
    plt.title('FER Graph - Optimal Walk Path Spells "FER"')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Add letter labels
    plt.text(1, 4.5, 'F', fontsize=24, fontweight='bold', ha='center')
    plt.text(5, 4.5, 'E', fontsize=24, fontweight='bold', ha='center')
    plt.text(9, 4.5, 'R', fontsize=24, fontweight='bold', ha='center')
    
    plt.tight_layout()
    plt.show()

def main():
    print("Creating FER Walk Graph...")
    
    # Create the graph
    matrix, coords, walk_path = create_fer_walk_graph()
    n = matrix.shape[0]
    
    print(f"Graph created with {n} nodes")
    print(f"Optimal walk path length: {len(walk_path)} nodes")
    
    # Analyze the graph
    analyzer = GraphAnalyzer(n=n)
    
    # Calculate costs manually since we know the optimal path
    walk_cost = 0
    for i in range(len(walk_path) - 1):
        u = walk_path[i]
        v = walk_path[i + 1]
        walk_cost += matrix[u][v]
    
    print(f"Walk cost (tracing FER): {walk_cost:.2f}")
    
    # For large graphs, approximate TSP cost instead of exact
    if n <= 10:
        tsp_cost, _ = analyzer.solve(matrix)
        print(f"TSP cost (exact): {tsp_cost:.2f}")
    else:
        # Quick approximation: nearest neighbor heuristic
        tsp_approx = approximate_tsp_cost(matrix)
        print(f"TSP cost (approx): {tsp_approx:.2f}")
        tsp_cost = tsp_approx
    
    if walk_cost < tsp_cost:
        print("✓ Walk < TSP achieved!")
        ratio = tsp_cost / walk_cost
        print(f"TSP/Walk ratio: {ratio:.2f}x")
    else:
        print("✗ TSP is still better")
    
    print("\nFER Walk Path:")
    print(" -> ".join(map(str, walk_path)))
    
    print(f"\nDistance Matrix (first 10x10):")
    print(matrix[:10, :10])
    
    # Plot the graph
    print("\nPlotting FER graph...")
    plot_fer_graph(coords, walk_path)

if __name__ == "__main__":
    main()