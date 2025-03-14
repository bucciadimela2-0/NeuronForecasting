import numpy as np

class DataHandler:
    @staticmethod
    def generate_adjacency_matrix(base_matrix, n):
        G = base_matrix.copy()
        for _ in range(n - 1):
            G = np.kron(G, base_matrix)
        np.fill_diagonal(G, 0)
        return G / np.mean(G[G > 0])