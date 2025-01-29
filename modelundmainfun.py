import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from mkl_random.mklrand import geometric
from torch.utils.data import DataLoader, random_split
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GraphConv
import torch.optim as optim
from torch.nn import Linear
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric_temporal.nn import TGCN
from Mydataset import SkeletonDataset
from clean_load import get_data_to_list
# from torch_geometric.utils import add_self_loops


##
# Beispiel-Daten

dataset = SkeletonDataset(torch.utils.data.Dataset)

indices = torch.randperm(len(dataset)).tolist()
dataset_tr = torch.utils.data.Subset(dataset, indices[:4000])
dataset_tes = torch.utils.data.Subset(dataset, indices[-551:])
#oder:
"""
dataset_size = len(dataset)
test_size = 551  # Größe des Testsets
train_size = dataset_size - test_size  # Rest für das Trainingsset

# Zufällige Teilung der Daten
dataset_tr, dataset_tes = random_split(dataset, [train_size, test_size])
"""

data_loader = DataLoader(dataset_tr, shuffle=True)
data_loader_test = DataLoader(dataset_tes, shuffle=False)
##

# in = data_loader.x

# Modell initialisieren
model = TGCN(in_channels=3, out_channels=45)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()


# def add_self_loops(edge_index, num_nodes):
#     #füge Self-Loops zu jedem Graphen im Batch hinzu, jeder Knoten eine Schleife auf sich selbst bekommt.
#     edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
#
#     return edge_index


edge_index = (torch.tensor([
    [0, 0], [0, 1], [1, 2], [2, 3], [3, 4],  # Daumen
    [0, 5], [5, 6], [6, 7], [7, 8],  # Zeigefinger
    [0, 9], [9, 10], [10, 11], [11, 12],  # Mittelfinger
    [0, 13], [13, 14], [14, 15], [15, 16],  # Ringfinger
    [0, 17], [17, 18], [18, 19], [19, 20]  # Kleiner Finger
], dtype=torch.long).t())

# edge_index = add_batch_self_loops(edge_indx, 21)  # Fügt Self-Loops hinzu

def train():
    model.train()

    for data in data_loader:  # Iterate in batches over the training dataset.
         out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
         out = global_mean_pool(out, data.batch)  # (1, 45)
         loss = criterion(out, data.y.view(-1))  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.

def test(loader):
     model.eval()

     correct = 0
     for data in data_loader_test:  # Iterate in batches over the training/test dataset.

         # edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.x.shape[0])

         out = model(data.x, data.edge_index, data.batch)
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.


for epoch in range(1, 10000):
    train()
    train_acc = test(data_loader)
    test_acc = test(data_loader_test)

    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    # Adapt the learning rate based on the epoch    #Weight Decay?
    if epoch == 50:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.1 * param_group['lr']


# def predict(model, image):
#     model.eval()
#     with torch.no_grad():
#         image = transform(image).unsqueeze(0).cuda()  # Batch-Dimension hinzufügen
#         outputs = model(image)
#         return outputs.cpu().numpy().flatten()  # Gibt die XYZ-Koordinaten der Keypoints zurück

