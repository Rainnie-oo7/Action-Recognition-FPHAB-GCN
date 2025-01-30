import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
import cv2
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
from load_all_skeletons import load_all_skeletons, load_data_from_skeleton_path

class SkeletonDataset(Dataset):
    def __init__(self, _):
        path = osp.normpath(osp.join(osp.dirname(__file__), "First-PersonHandActionBenchmarkF-PHAB"))

        # Lädt pro Action die Skelett-Pfade
        self.skeleton_paths, self.actions = load_all_skeletons(path)
        self.hotend_indexed_dict = {action: index for index, action in enumerate(self.actions)}
        # Lädt pro Action die Skelette / Data
        self.datatl = load_data_from_skeleton_path(self.skeleton_paths, self.hotend_indexed_dict)

    def __len__(self):
        return len(self.datatl)

    def __getitem__(self, idx):

        edge_index = torch.tensor([
            [0, 1], [1, 2], [2, 3], [3, 4],  # Daumen
            [0, 5], [5, 6], [6, 7], [7, 8],  # Zeigefinger
            [0, 9], [9, 10], [10, 11], [11, 12],  # Mittelfinger
            [0, 13], [13, 14], [14, 15], [15, 16],  # Ringfinger
            [0, 17], [17, 18], [18, 19], [19, 20]  # Kleiner Finger
        ], dtype=torch.long)

        node_features = torch.tensor(self.datatl[idx][1], dtype=torch.float32)
        data = Data(x=node_features, edge_index=edge_index.t(), y = torch.tensor(self.datatl[idx][0], dtype=torch.uint8))
        data.y = data.y.clamp(0, 20)    # für cross entropy; braucht einen id weniger

        return data




