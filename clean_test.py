# import Mydataset
# from Mydataset import SkeletonDataset
# import torch
import os.path as osp
# import matplotlib.pyplot as plt
# import numpy as np
# import cv2
from load_all_skeletons import load_all_skeletons, load_data_from_skeleton_path


if __name__ == '__main__':
    path = osp.normpath(osp.join(osp.dirname(__file__), "First-PersonHandActionBenchmarkF-PHAB"))
    # dataset = SkeletonDataset(path)
    # idx = 17
    # # print(dataset.imgs[idx])
    # print(dataset[idx])
    skeleton_paths = load_all_skeletons(path)
    print()
    print()