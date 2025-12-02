# mzwv_solvers.py
import itertools
import networkx as nx
import matplotlib.pyplot as plt
from pulp import (
    LpProblem, LpVariable, LpMinimize, lpSum, LpStatus, value, PULP_CBC_CMD
)


class Model:
    def __init__(self, name, version, distance_matrix=None):
        self.name = name
        self.version = version
        self.distance_matrix = distance_matrix
        # build directed graph from distance matrix (complete digraph with weights)
        self._build_graph_from_distance_matrix()

    def _build_graph_from_distance_matrix(self):
        self.graph = nx.DiGraph()
        n = len(self.distance_matrix)
        for i in range(n):
            self.graph.add_node(i)
        for i in range(n):
            for j in range(n):
                if i != j:
                    w = self.distance_matrix[i][j]
                    # assume non-negative finite weights; if inf or None, skip
                    self.graph.add_edge(i, j, weight=w)

    def get_info(self):
        return f"Model Name: {self.name}, Version: {self.version}"

    def get_distance_matrix(self):
        return self.distance_matrix

    def plot_graph(self, route_nodes=None, title=None):
        """
        Draw the underlying directed graph (weights) and optionally highlight
        a route given as a list of nodes in visit order (walk nodes).
        """
        G = self.graph
        pos = nx.spring_layout(G, seed=1)
        plt.figure(figsize=(8, 6))
        # draw nodes and all edges lightly
        nx.draw_networkx_nodes(G, pos, node_size=300)
        nx.draw_networkx_labels(G, pos)
        # draw all edges faint
        all_edges = list(G.edges())
        nx.draw_networkx_edges(G, pos, edgelist=all_edges, alpha=0.2, arrows=True)
        # edge labels
        weights = {(u, v): f"{G[u][v]['weight']}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=weights, font_size=8)

        # if route provided, highlight it as a sequence of edges
        if route_nodes:
            # route_nodes is a list of nodes in order (walk)
            route_edges = []
            for a, b in zip(route_nodes, route_nodes[1:]):
                route_edges.append((a, b))
            nx.draw_networkx_edges(
                G, pos, edgelist=route_edges, edge_color="r", width=2.5, arrows=True
            )
        if title:
            plt.title(title)
        plt.axis("off")
        plt.show()

    def best_path_of_perumtation(self, perm):
        """
        Given a permutation of nodes, compute the best path cost
        by summing shortest paths between consecutive nodes in the perm.
        Returns total_cost, walk_nodes (full sequence including returns)
        """
        G = self.graph
        total_cost = 0.0
        walk_nodes = []
        feasible = True
        for i in range(len(perm) - 1):
            a = perm[i]
            b = perm[i + 1]
            try:
                path = nx.shortest_path(G, a, b, weight="weight")
                cost = nx.shortest_path_length(G, a, b, weight="weight")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                feasible = False
                break
            walk_nodes.extend(path[:-1])  # append all but last to chain
            total_cost += cost
        if not feasible:
            return float("inf"), []
        # finally return to start
        try:
            path = nx.shortest_path(G, perm[-1], perm[0], weight="weight")
            cost = nx.shortest_path_length(G, perm[-1], perm[0], weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return float("inf"), []
        walk_nodes.extend(path)  # include last->start path fully
        total_cost += cost

        return walk_nodes, total_cost

    def brute_force_tsp(self):
        
        """
        Brute-force TSP on complete graph using distance matrix.
        Returns best_permutation, best_cost
        """
        n = len(self.distance_matrix)
        nodes = list(range(n))

        best_cost = float("inf")
        best_perm = None

        for perm in itertools.permutations(nodes):
            total_cost = 0.0
            for i in range(n):
                a = perm[i]
                b = perm[(i + 1) % n]  # wrap around to start
                total_cost += self.distance_matrix[a][b]
            if total_cost < best_cost:
                best_cost = total_cost
                best_perm = perm

        walkNodes, cost = self.best_path_of_perumtation(best_perm)
        print("Best walk nodes from brute-force TSP:", walkNodes, "with cost:", cost)
        return best_perm, best_cost

        
    # -----------------------------
    # BRUTE-FORCE (permutations + shortest paths between consecutive)
    # -----------------------------
    def brute_force(self, start_node=0):
        """
        Brute-force: iterate all permutations of nodes (fixing start to reduce symmetries),
        between permuted nodes use shortest path in the underlying graph.
        Returns: best_permutation, best_walk_nodes (sequence including returns), best_cost
        """
        G = self.graph
        n = G.number_of_nodes()
        nodes = list(G.nodes())
        if start_node not in nodes:
            raise ValueError("start_node must be in graph nodes")

        # fix start node as first element in permutation to avoid circular duplicates
        other_nodes = [v for v in nodes if v != start_node]

        best_cost = float("inf")
        best_perm = None
        best_walk = None

        for perm in itertools.permutations(other_nodes):

            order = (start_node,) + perm
            total_cost = 0.0
            walk_nodes = []
            feasible = True
            # between consecutive in order, append shortest path (except last node to keep chaining)
            for i in range(len(order) - 1):
                a = order[i]
                b = order[i + 1]
                try:
                    path = nx.shortest_path(G, a, b, weight="weight")
                    cost = nx.shortest_path_length(G, a, b, weight="weight")
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    feasible = False
                    break
                # append all but last to chain
                walk_nodes.extend(path[:-1])
                total_cost += cost
            if not feasible:
                continue
            # finally return to start
            try:
                path = nx.shortest_path(G, order[-1], start_node, weight="weight")
                cost = nx.shortest_path_length(G, order[-1], start_node, weight="weight")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            walk_nodes.extend(path)  # include last->start path fully
            total_cost += cost

            # sanity: ensure all nodes visited at least once
            visited = set(walk_nodes)
            # include final start node
            if start_node not in visited:
                visited.add(start_node)
            if set(nodes) - visited:
                # some node not visited -> shouldn't happen if shortest paths cover nodes,
                # but keep safety check
                continue

            if total_cost < best_cost:
                best_cost = total_cost
                best_perm = order
                best_walk = walk_nodes

        return best_perm, best_walk, best_cost

    # -----------------------------
    # Helper: compute condensed distance matrix (all-pairs shortest path lengths)
    # -----------------------------
    def compute_condensed_distances(self):
        """
        Compute shortest-path distances between every pair of graph nodes
        in self.graph and return a matrix dist[i][j].
        """
        G = self.graph
        n = G.number_of_nodes()
        nodes = list(G.nodes())
        dist = [[float("inf")] * n for _ in range(n)]
        for i, u in enumerate(nodes):
            # use Dijkstra from u
            length = nx.single_source_dijkstra_path_length(G, u, weight="weight")
            for j, v in enumerate(nodes):
                if u == v:
                    dist[i][j] = 0.0
                else:
                    if v in length:
                        dist[i][j] = float(length[v])
                    else:
                        dist[i][j] = float("inf")
        return dist

    # -----------------------------
    # MTZ MILP on condensed complete graph (TSP on dist matrix)
    # -----------------------------
    def milp_mtz(self, start_node=0, solver_cmd=None):
        """
        MTZ formulation for TSP on condensed graph where costs are shortest-path distances.
        Returns permutation order, reconstructed_walk (nodes along original graph), total_cost.
        """
        # condensed distances
        dist = self.compute_condensed_distances()
        n = len(dist)
        nodes = list(range(n))
        # MTZ requires no infinite distances; we ensure graph is strongly connected for feasibility
        # But to be safe, check for inf
        for i in range(n):
            for j in range(n):
                if i != j and dist[i][j] == float("inf"):
                    raise ValueError("Condensed graph not strongly connected, some pairs have no path")

        # build MILP
        prob = LpProblem("MTZ_TSP", LpMinimize)
        # x[i,j] binary for i != j
        x = LpVariable.dicts("x", ((i, j) for i in nodes for j in nodes if i != j), cat="Binary")
        # u vars for MTZ (only for 1..n-1)
        u = LpVariable.dicts("u", (i for i in nodes), lowBound=0, upBound=n - 1, cat="Integer")

        # objective
        prob += lpSum(dist[i][j] * x[i, j] for i in nodes for j in nodes if i != j)

        # degree constraints
        for k in nodes:
            prob += lpSum(x[i, k] for i in nodes if i != k) == 1
            prob += lpSum(x[k, j] for j in nodes if j != k) == 1

        # MTZ subtour elimination
        # standard: for i != j and i != 0 and j != 0:
        for i in nodes:
            for j in nodes:
                if i != j and i != start_node and j != start_node:
                    prob += u[i] - u[j] + (n - 1) * x[i, j] <= n - 2

        # fix u[start_node] = 0 for stability (optional)
        prob += u[start_node] == 0

        # solve
        if solver_cmd is None:
            solver_cmd = PULP_CBC_CMD(msg=False)
        prob.solve(solver_cmd)

        status = LpStatus[prob.status]
        if status != "Optimal":
            # return status info
            return None, None, None, status

        # extract tour order from x
        succ = {}
        for i in nodes:
            for j in nodes:
                if i != j and value(x[i, j]) > 0.5:
                    succ[i] = j
        # rebuild tour starting from start_node
        tour = [start_node]
        cur = start_node
        for _ in range(n - 1):
            cur = succ[cur]
            tour.append(cur)
        # confirm returns to start implicitly
        # reconstruct walk in original graph by concatenating shortest paths
        walk_nodes = []
        total_cost = 0.0
        for i in range(len(tour) - 1):
            a, b = tour[i], tour[i + 1]
            path = nx.shortest_path(self.graph, a, b, weight="weight")
            cost = nx.shortest_path_length(self.graph, a, b, weight="weight")
            walk_nodes.extend(path[:-1])
            total_cost += cost
        # add final return to start
        path = nx.shortest_path(self.graph, tour[-1], start_node, weight="weight")
        cost = nx.shortest_path_length(self.graph, tour[-1], start_node, weight="weight")
        walk_nodes.extend(path)
        total_cost += cost

        return tour, walk_nodes, total_cost, status

    # -----------------------------
    # Flow-based subtour-elimination MILP on condensed graph
    # (classical single-commodity flow formulation)
    # -----------------------------
    def milp_flow(self, start_node=0, solver_cmd=None):
        """
        Flow-based subtour elimination TSP on condensed graph (costs = shortest path distances).
        This returns a tour on condensed nodes that can be unfolded into a walk on original graph.
        """
        dist = self.compute_condensed_distances()
        n = len(dist)
        nodes = list(range(n))
        for i in range(n):
            for j in range(n):
                if i != j and dist[i][j] == float("inf"):
                    raise ValueError("Condensed graph not strongly connected, some pairs have no path")

        prob = LpProblem("Flow_TSP", LpMinimize)
        x = LpVariable.dicts("x", ((i, j) for i in nodes for j in nodes if i != j), cat="Binary")
        # flow variables: continuous >=0
        f = LpVariable.dicts("f", ((i, j) for i in nodes for j in nodes if i != j), lowBound=0)

        # objective
        prob += lpSum(dist[i][j] * x[i, j] for i in nodes for j in nodes if i != j)

        # degree constraints
        for k in nodes:
            prob += lpSum(x[i, k] for i in nodes if i != k) == 1
            prob += lpSum(x[k, j] for j in nodes if j != k) == 1

        # flow constraints: source = start_node sends out (n-1) units
        prob += lpSum(f[start_node, j] for j in nodes if j != start_node) == n - 1

        # for all other nodes: inflow - outflow = 1 (they consume one unit)
        for v in nodes:
            if v == start_node:
                continue
            prob += lpSum(f[i, v] for i in nodes if i != v) - lpSum(f[v, j] for j in nodes if j != v) == 1

        # flow coupling: f[i,j] <= (n-1) * x[i,j]
        for i in nodes:
            for j in nodes:
                if i != j:
                    prob += f[i, j] <= (n - 1) * x[i, j]

        # solve
        if solver_cmd is None:
            solver_cmd = PULP_CBC_CMD(msg=False)
        prob.solve(solver_cmd)

        status = LpStatus[prob.status]
        if status != "Optimal":
            return None, None, None, status

        succ = {}
        for i in nodes:
            for j in nodes:
                if i != j and value(x[i, j]) > 0.5:
                    succ[i] = j
        # reconstruct tour
        tour = [start_node]
        cur = start_node
        for _ in range(n - 1):
            cur = succ[cur]
            tour.append(cur)

        # reconstruct full walk on original graph
        walk_nodes = []
        total_cost = 0.0
        for i in range(len(tour) - 1):
            a, b = tour[i], tour[i + 1]
            path = nx.shortest_path(self.graph, a, b, weight="weight")
            cost = nx.shortest_path_length(self.graph, a, b, weight="weight")
            walk_nodes.extend(path[:-1])
            total_cost += cost
        # return to start
        path = nx.shortest_path(self.graph, tour[-1], start_node, weight="weight")
        cost = nx.shortest_path_length(self.graph, tour[-1], start_node, weight="weight")
        walk_nodes.extend(path)
        total_cost += cost

        return tour, walk_nodes, total_cost, status

import numpy as np
import time
from MetricFriendlyGraphGenerator import MetricFriendlyGraphGenerator
def main():
    # example distance matrix (complete directed with weights)
    distance_matrix = np.array([
        [0, 1, 2, 1],
        [1, 0, 1, 3],
        [2, 1, 0, 4],
        [1, 3, 4, 0],
    ])

    n =5
    #np.random.seed(2)

    #distance_matrix = np.random.rand(n, n) * 10
    #distance_matrix = (distance_matrix + distance_matrix.T) / 2
    
    np.fill_diagonal(distance_matrix, 0)

    model = Model("DistanceModel", "1.0", distance_matrix=distance_matrix)

    print(distance_matrix)
    # brute force
    print("\nRunning brute-force (permute order + shortest paths)...")
    start = time.time()
    bf_perm, bf_walk, bf_cost = model.brute_force(start_node=0)
    endTime = time.time() - start
    print("Brute-force elapsed time (s):", endTime)
    print("Brute-force perm order:", bf_perm)
    print("Brute-force walk nodes:", bf_walk)
    print("Brute-force total cost:", bf_cost)

    print("\n running brute-force TSP on complete graph...")
    start = time.time()
    bf_tsp_perm, bf_tsp_cost = model.brute_force_tsp()
    endTime = time.time() - start
    print("Brute-force TSP elapsed time (s):", endTime)
    print("Brute-force TSP perm order:", bf_tsp_perm)
    print("Brute-force TSP total cost:", bf_tsp_cost)

    # MTZ MILP
    print("\nRunning MTZ MILP (condensed graph TSP)...")
    start = time.time()
    mtz_tour, mtz_walk, mtz_cost, mtz_status = model.milp_mtz(start_node=0)
    endTime = time.time() - start
    print("MTZ elapsed time (s):", endTime)
    print("MTZ status:", mtz_status)
    print("MTZ tour (condensed nodes):", mtz_tour)
    print("MTZ full walk nodes:", mtz_walk)
    print("MTZ total cost:", mtz_cost)

    # Flow MILP
    print("\nRunning Flow-based MILP (single-commodity flow subtour elimination)...")
    start = time.time()
    flow_tour, flow_walk, flow_cost, flow_status = model.milp_flow(start_node=0)
    endTime = time.time() - start
    print("Flow elapsed time (s):", endTime)
    print("Flow status:", flow_status)
    print("Flow tour (condensed nodes):", flow_tour)
    print("Flow full walk nodes:", flow_walk)
    print("Flow total cost:", flow_cost)

    # plot graph with MTZ route highlighted (if exists)
    #model.plot_graph(route_nodes=bf_tsp_perm, title=f"Brute-force walk (cost={bf_tsp_cost})")

    #model.plot_graph(route_nodes=mtz_walk, title=f"MTZ walk (cost={mtz_cost})")



if __name__ == "__main__":
    main()
