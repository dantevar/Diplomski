import random
import itertools
import math
import numpy as np

class MetricFriendlyGraphGenerator:
    """
    Generira grafe kod kojih je optimalni metric-closure TSP ciklus
    gotovo uvijek MANJI od optimalnog TSP-a s direktnim edge težinama.

    Mehanizam:
    1) Radi klastere čvorova s vrlo jeftinim internim bridovima
    2) Između klastera postavlja ekstremno skupe bridove
    3) Kasnije shortest-path (metric closure) obara cijene između klastera,
       jer umjesto direktnog debelog brida koristi chain kroz hub klaster.
    """

    def __init__(self, 
                 n=10, 
                 num_clusters=3,
                 intra_low=1,
                 intra_high=5,
                 inter_low=50,
                 inter_high=200):
        self.n = n
        self.num_clusters = num_clusters
        self.intra_low = intra_low
        self.intra_high = intra_high
        self.inter_low = inter_low
        self.inter_high = inter_high

    def generate(self):
        # 1) Podijeli čvorove u klastere
        clusters = [[] for _ in range(self.num_clusters)]
        for i in range(self.n):
            clusters[i % self.num_clusters].append(i)

        # 2) Kreiraj praznu matricu
        M = np.zeros((self.n, self.n), dtype=float)

        # 3) Unutar klastera: JEFTINO
        for c in clusters:
            for u, v in itertools.permutations(c, 2):
                M[u][v] = random.uniform(self.intra_low, self.intra_high)

        # 4) Između klastera: SKUPO
        for c1, c2 in itertools.permutations(range(self.num_clusters), 2):
            for u in clusters[c1]:
                for v in clusters[c2]:
                    # pazi da NE krši trokutnu na "pozitivan" način
                    # nego da ga razbije – forsa velike edge costove
                    M[u][v] = random.uniform(self.inter_low, self.inter_high)

        # 5) Dijagonala 0
        for i in range(self.n):
            M[i][i] = 0

        return M
