from load_all_skeletons import load_all_skeletons, load_data_from_skeleton_path
import os.path as osp
import torch
from torch_geometric.data import Data, Batch
import numpy as np


def get_data_to_list(path):
    # Lädt pro Action die Skelett-Pfade
    skeleton_paths, actions = load_all_skeletons(path)
    hotend_indexed_dict = {action: index for index, action in enumerate(actions)}
    # Lädt pro Action die Skelette / Data
    datatl = load_data_from_skeleton_path(skeleton_paths, hotend_indexed_dict)
    print()
    return datatl
#Edge_index und Adjazenzmatrix
# edge_index = torch.tensor([
#     [0, 1], [1, 2], [2, 3], [3, 4],  # Daumen
#     [0, 5], [5, 6], [6, 7], [7, 8],  # Zeigefinger
#     [0, 9], [9, 10], [10, 11], [11, 12],  # Mittelfinger
#     [0, 13], [13, 14], [14, 15], [15, 16],  # Ringfinger
#     [0, 17], [17, 18], [18, 19], [19, 20]  # Kleiner Finger
# ], dtype=torch.long)

#
# def adjacency_matrix(edges, num_nodes):
#     A = np.zeros((num_nodes, num_nodes), dtype=int)
#     for u, v in edges:
#         A[u, v] = 1
#         A[v, u] = 1  # Because it's an undirected graph
#     return A
#
# edges = [
#     (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
#     (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
#     (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
#     (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
#     (0, 17), (17, 18), (18, 19), (19, 20)  # Little finger
# ]
#
# num_nodes = 21  # Since nodes are indexed from 0 to 20

if __name__ == '__main__':
    path = osp.normpath(osp.join(osp.dirname(__file__), "First-PersonHandActionBenchmarkF-PHAB"))
    # adj_matrix = adjacency_matrix(edges, num_nodes)
    # print(adj_matrix)
    datatl = get_data_to_list(path)
    print()