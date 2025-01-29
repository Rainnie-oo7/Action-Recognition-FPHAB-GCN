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

class TGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TGCN, self).__init__()
        self.recurrent = TGCN(in_channels, out_channels, 2)
        self.linear = torch.nn.Linear(out_channels, 45)

    def forward(self, x, edge_index):
        h = self.recurrent(x, edge_index)
        h = F.relu(h)
        h = self.linear(h)
        return h


##
# Beispiel-Daten
input_size = 63
hidden_size = 256
num_layers = 1
num_classes = 45

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

data_loader = torch.utils.data.DataLoader(dataset_tr, batch_size=1, shuffle=True)
data_loader_test = torch.utils.data.DataLoader(dataset_tes, batch_size=1, shuffle=False)
##

# in = data_loader.x

# Modell initialisieren
model = TGCN(in_channels=63, out_channels=63)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    for data in data_loader:  # Iterate in batches over the training dataset.
         out = model(data.x, data.edge_index)  # Perform a single forward pass.
         loss = criterion(out, data.y.view(-1))  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.

def test(loader):
     model.eval()

     correct = 0
     for data in data_loader_test:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index)
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

