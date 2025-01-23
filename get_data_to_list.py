from load_all_skeletons import load_all_skeletons, load_data_from_skeleton_path
import os.path as osp
import torch
from torch_geometric.data import Data, Batch



def get_data_to_list(path):
    # Lädt pro Action die Skelett-Pfade
    skeleton_paths, actions = load_all_skeletons(path)
    hotend_indexed_dict = {action: index for index, action in enumerate(actions)}
    # Lädt pro Action die Skelette / Data
    datatl = load_data_from_skeleton_path(skeleton_paths, hotend_indexed_dict)
    print()


    train_data = []

    for idx in range(len(datatl)):
        edge_index = torch.tensor([
            [0, 1], [1, 2], [2, 3], [3, 4],  # Daumen
            [0, 5], [5, 6], [6, 7], [7, 8],  # Zeigefinger
            [0, 9], [9, 10], [10, 11], [11, 12],  # Mittelfinger
            [0, 13], [13, 14], [14, 15], [15, 16],  # Ringfinger
            [0, 17], [17, 18], [18, 19], [19, 20]  # Kleiner Finger
        ], dtype=torch.long)
        node_features = torch.tensor(datatl[idx][1], dtype=torch.float32)
        data = Data(x=node_features, edge_index=edge_index.t(),
                    y=torch.tensor(datatl[idx][0], dtype=torch.uint8))
        data.y = data.y.clamp(0, 20)  # für cross entropy; braucht einen id weniger
        train_data.append(data)
    return train_data