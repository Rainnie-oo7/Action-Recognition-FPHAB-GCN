import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
from torch.utils.data import Dataset, DataLoader
import load_clean
import load_all_skeletons
import process_skeleton as sk
from configs import config_seamer as cf

sys.path.append('/home/boris.grillborzer/PycharmProjects/PoseEstimationFPHAB')
data, _, _ = sk.process_files(cf.path_dataset)
path_dataset_depthimages = cf.path_dataset_depthimages


class SkeletonDataset(Dataset):
    def __init__(self, path_dataset_depthimages, data_paths, labels, transform=None):
        for dir in path_dataset_depthimages:
            folder = os.join
        self.image_paths = image_paths  # Liste von Pfaden
        self.keypoint_annotations = keypoint_annotations
        self.labels = ["SpineBase", "SpineMid", "Neck", "Head", "ShoulderLeft", "ElbowLeft",
                       "WristLeft", "HandLeft", "ShoulderRight", "ElbowRight", "WristRight",
                       "HandRight", "HipLeft", "KneeLeft", "AnkleLeft", "FootLeft",
                       "HipRight", "KneeRight", "AnkleRight", "FootRight", "SpineShoulder",
                       "HandTipLeft", "ThumbLeft", "HandTipRight", "ThumbRight"]  # 25 Stück
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Von BGR zu RGB konvertieren

        # Lade die Skelettdaten
        keypoints = np.loadtxt(self.data_paths[idx])  # Beispiel für das Laden der txt-Daten
        label = self.labels[idx]

        # Optional: Transformiere die Daten (z. B. Normalisierung)
        if self.transform:
            data = self.transform(data)

        return data, label


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




