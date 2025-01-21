import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
import cv2
from torch.utils.data import Dataset, DataLoader
from load_all_skeletons import load_all_skeletons, load_data_from_skeleton_path


class SkeletonDataset(Dataset):
    def __init__(self, path, transform=None):
        # Lädt pro Action die Skelett-Pfade
        self.skeleton_paths, self.actions = load_all_skeletons(path)
        # Lädt pro Action die Skelette / Data
        self.skeleton_data = load_data_from_skeleton_path(self.skeleton_paths)
        self.labels = ["Wrist, TMCP, IMCP, MMCP, RMCP, PMCP, TPIP, TDIP, TTIP, IPIP, IDIP, ITIP, MPIP, MDIP, MTIP, RPIP, RDIP, RTIP, PPIP, PDIP, PTIP"]  # 21 Stück
        self.transform = transform

    def __len__(self):
        return len(self.skeleton_data['charge_cell_phone'])

    def __getitem__(self, idx):

        edge_index = torch.tensor([  # Adjazenzmatrix für 21 Handgelenke (Indexierung korrigiert)
            (0, 1), (1, 2), (2, 3), (3, 4),  # Daumen
            (0, 5), (5, 6), (6, 7), (7, 8),  # Zeigefinger
            (0, 9), (9, 10), (10, 11), (11, 12),  # Mittelfinger
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ringfinger
            (0, 17), (17, 18), (18, 19), (19, 20)  # Kleiner Finger
        ], dtype=torch.long)

        y = self.actions    # 45 Klassen
        for i in self.actions:
            if i == 'charge_cell_phone':
                y = i

        #The shape of node_features should be
        # (num_nodes, num_features),
        # 21, 3
        # where num_nodes is the total number of nodes in the graph, and num_features is the number of features each node has
        node_features = self.skeleton_data['charge_cell_phone'][idx]
        # Optional: Transformiere die Daten (z. B. Normalisierung)
        if self.transform:
            node_features = self.transform(node_features)

        return node_features, edge_index, y




