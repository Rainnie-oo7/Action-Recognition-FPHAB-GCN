import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
from torch.utils.data import Dataset, DataLoader
from load_all_skeletons import load_all_skeletons, load_data_from_skeleton_path

class SkeletonDataset(Dataset):
    def __init__(self, path_dataset_depthimages, data_paths, labels, transform=None):

        self.image_paths = image_paths  # Liste von Pfaden
        self.keypoint_annotations = keypoint_annotations
        self.labels = ["Wrist, TMCP, IMCP, MMCP, RMCP, PMCP, TPIP, TDIP, TTIP, IPIP, IDIP, ITIP, MPIP, MDIP, MTIP, RPIP, RDIP, RTIP, PPIP, PDIP, PTIP"]  # 21 Stück
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Von BGR zu RGB konvertieren


        # Optional: Transformiere die Daten (z. B. Normalisierung)
        if self.transform:
            data = self.transform(data)

        return ???


data_root = osp.normpath(
    osp.join(osp.dirname(__file__), "First-PersonHandActionBenchmarkF-PHAB\Hand_pose_annotation_v1"))
skeleton_paths = load_all_skeletons.load_all_skeletons(data_root)
skeleton_data = load_all_skeletons.load_data_from_skeleton_path(skeleton_paths)


def mkthreesome():
    # Jede Drei Koordntn zu ein Label zuordnen
    results = {}

    for action, spatvals in skeleton_data.items():

        # Wähle drei Werte für jedes der 21-3er Labeltripel
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




