import Mydataset
from Mydataset import SkeletonDataset
import torch
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
from load_all_skeletons import load_all_skeletons, load_data_from_skeleton_path


if __name__ == '__main__':
    path = osp.normpath(osp.join(osp.dirname(__file__), "First-PersonHandActionBenchmarkF-PHAB"))
    dataset = SkeletonDataset(path)
    idx = 17
    # print(dataset.imgs[idx])
    print(dataset[idx])

    #Lädt pro Action die Skelett-Pfade
    # skeleton_paths, actions = load_all_skeletons(path)
    #Lädt pro Action die Skelette / Data
    # skeleton_data, grouped_data_list = load_data_from_skeleton_path(skeleton_paths)

    # data = np.random.rand(79, 63)  # Zufällige Werte als Beispiel
    # # gd = bobtest(data)
    print()
    print()