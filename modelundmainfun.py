import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import TGCN
from torch_geometric_temporal.signal import temporal_signal_split
from Mydataset import StaticGraphTemporalSignal
from load_all_skeletons import load_all_skeletons, load_data_from_skeleton_path

path = osp.normpath(osp.join(osp.dirname(__file__), "First-PersonHandActionBenchmarkF-PHAB"))

# Lädt pro Action die Skelett-Pfade
skeleton_paths, actions = load_all_skeletons(path)
hotend_indexed_dict = {action: index for index, action in enumerate(actions)}
# Lädt pro Action die Skelette / Data
labels, datatl = load_data_from_skeleton_path(skeleton_paths, hotend_indexed_dict)
print()
edge_index = np.transpose(np.array([[0, 1], [1, 2], [2, 3], [3, 4],  # Daumen
             [0, 5], [5, 6], [6, 7], [7, 8],  # Zeigefinger
             [0, 9], [9, 10], [10, 11], [11, 12],  # Mittelfinger
             [0, 13], [13, 14], [14, 15], [15, 16],  # Ringfinger
             [0, 17], [17, 18], [18, 19], [19, 20]])) # Kleiner Finger

# loader = StaticGraphTemporalSignal(,
##
# Beispiel-Daten
dataset = StaticGraphTemporalSignal(edge_index=edge_index,
                                    features= datatl,
                                    targets= labels,
                                    edge_weight=None,
                                    )
print()
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable):
        return iterable

train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.2)

class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(RecurrentGCN, self).__init__()
        self.recurrent = TGCN(node_features, 32)
        self.linear = torch.nn.Linear(32, 45)

    def forward(self, x, edge_index, edge_weight, prev_hidden_state):
        h = self.recurrent(x, edge_index, edge_weight, prev_hidden_state)
        y = F.relu(h)
        y = self.linear(y)
        return y, h


model = RecurrentGCN(node_features=63)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()

for epoch in tqdm(range(50)):
    cost = 0
    hidden_state = None
    for time, snapshot in enumerate(train_dataset):
        y_hat, hidden_state = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr, hidden_state)
        cost = cost + torch.mean((y_hat - snapshot.y) ** 2)
    cost = cost / (time + 1)
    cost.backward()
    optimizer.step()
    optimizer.zero_grad()

model.eval()
cost = 0
hidden_state = None
for time, snapshot in enumerate(test_dataset):
    y_hat, hidden_state = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr, hidden_state)
    cost = cost + torch.mean((y_hat - snapshot.y) ** 2)
cost = cost / (time + 1)
cost = cost.item()
print("MSE: {:.4f}".format(cost))
