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
        self.labels = ["Wrist, TMCP, IMCP, MMCP, RMCP, PMCP, TPIP, TDIP, TTIP, IPIP, IDIP, ITIP, MPIP, MDIP, MTIP, RPIP, RDIP, RTIP, PPIP, PDIP, PTIP"]  # 21 Stück
        self.actions =
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Von BGR zu RGB konvertieren

        edge_index = torch.tensor([  # Adjazenzmatrix für 21 Handgelenke (Indexierung korrigiert)
            (0, 1), (1, 2), (2, 3), (3, 4),  # Daumen
            (0, 5), (5, 6), (6, 7), (7, 8),  # Zeigefinger
            (0, 9), (9, 10), (10, 11), (11, 12),  # Mittelfinger
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ringfinger
            (0, 17), (17, 18), (18, 19), (19, 20)  # Kleiner Finger
        ], dtype=torch.long)

        y = self.actions
        # Lädt pro Action die Skelette / Data
        skeleton_data = load_data_from_skeleton_path(self.skeleton_paths)
        # Optional: Transformiere die Daten (z. B. Normalisierung)
        if self.transform:
            data = self.transform(data)

        return node_features, y




