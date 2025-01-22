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
    def __init__(self, path, transform=None):
        # Lädt pro Action die Skelett-Pfade
        self.skeleton_paths, self.actions = load_all_skeletons(path)
        self.hotend_indexed_dict = {action: index for index, action in enumerate(self.actions)}
        # Lädt pro Action die Skelette / Data
        self.datatl = load_data_from_skeleton_path(self.skeleton_paths, self.hotend_indexed_dict)
        self.transform = transform

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

        # actions = [
        #     'use_flash', 'handshake', 'close_peanut_butter', 'light_candle', 'scoop_spoon', 'put_sugar',
        #     'wash_sponge', 'open_juice_bottle', 'prick', 'close_liquid_soap', 'flip_sponge', 'open_peanut_butter',
        #     'sprinkle', 'put_tea_bag', 'high_five', 'close_milk', 'pour_milk', 'give_card', 'open_milk',
        #     'squeeze_paper', 'flip_pages', 'give_coin', 'pour_juice_bottle', 'toast_wine', 'open_soda_can',
        #     'close_juice_bottle', 'put_salt', 'tear_paper', 'receive_coin', 'open_liquid_soap',
        #     'take_letter_from_enveloppe', 'pour_liquid_soap', 'unfold_glasses', 'charge_cell_phone', 'drink_mug',
        #     'open_letter', 'open_wallet', 'scratch_sponge', 'read_letter', 'pour_wine', 'write', 'clean_glasses',
        #     'squeeze_sponge', 'stir', 'use_calculator'
        # ]

        y = self.actions    # 45 Klassen


        #The shape of node_features should be
        # (num_nodes, num_features),
        # 21, 3
        # where num_nodes is the total number of nodes in the graph, and num_features is the number of features each node has

        node_features = torch.tensor(self.datatl[idx][1], dtype=torch.float32)
        data = Data(x=node_features.squeeze(), edge_index=edge_index.t(), y=torch.tensor(self.datatl[idx][0], dtype=torch.long))

        #Transformiere die Daten (z. B. Normalisierung)
        if self.transform:
            data = self.transform(data)

        print()
        return data




