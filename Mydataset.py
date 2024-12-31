import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np
from torch.utils.data import Dataset, DataLoader
import load_all_skeletons

class SkeletonDataset(Dataset):
    def __init__(self, data_paths, labels, transform=None):
        self.data_paths = data_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        # Lade die Skelettdaten
        data = np.loadtxt(self.data_paths[idx])  # Beispiel für das Laden der txt-Daten
        label = self.labels[idx]

        # Optional: Transformiere die Daten (z. B. Normalisierung)
        if self.transform:
            data = self.transform(data)

        return data, label

data_root = osp.normpath(osp.join(osp.dirname(__file__), "First-PersonHandActionBenchmarkF-PHAB\Hand_pose_annotation_v1"))
skeleton_paths = load_all_skeletons.load_all_skeletons(data_root)
skeleton_data = load_all_skeletons.load_data_from_skeleton_path(skeleton_paths)



# Liste mit 21 Labels
# labels = ['label_0','label_1','label_2','label_3','label_4','label_5','label_6','label_7','label_8','label_9','label_10','label_11',\
#           'label_12','label_13','label_14','label_15','label_16','label_17','label_18','label_19','label_20']  # Deine Gelenke
labels = [f"label_{i+1}" for i in range(21)]  # z. B. label_1, label_2, ...

# Jede Drei Koordntn zu ein Label zuordnen
results = {}

for action, spatvals in skeleton_data.items():

    # Wähle drei Werte für jedes der 21 Labels
    label_values = {}
    for i, label in enumerate(labels):
        if i < 64:
            label_values[label] = skeleton[i, :3]  # Die ersten drei Werte jeder Zeile
        # else:
        #     label_values[label] = np.array([None, None, None])  # Auffüllen, falls Daten fehlen

    # Speichere die Ergebnisse
    results[f"entry_{action}"] = label_values

# Ergebnisse ausgeben
for key, value in results.items():
    print(f"{key}:")
    for label, values in value.items():
        print(f"  {label}: {values}")


data_paths, _ = load_all_skeletons.load_all_skeletons(data_root)
dataset = SkeletonDataset(data_paths, labels)



