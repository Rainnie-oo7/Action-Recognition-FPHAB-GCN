import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from mkl_random.mklrand import geometric
from torch.utils.data import DataLoader, random_split
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GraphConv
import torch.optim as optim
from torch.nn import Linear
from torch_geometric.transforms import NormalizeFeatures

from Mydataset import SkeletonDataset
from clean_load import  get_data_to_list


##
## TODO: = x!!! Muss noch in richtige Shape gebracht werden (batch_size, sequence_length, nodes, features) 4551 Liste von Objekten á:  x, edgeindex, y
##


class TemporalGCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, num_nodes):
        super(TemporalGCN, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, y):
        # x = inn  # Node features

        gru_input = x.view(batch_size, seq_length, -1)  # Shape: (batch_size, seq_length, nodes * hidden_features)

        gru_out, _ = self.gru(gru_input)  # Shape: (batch_size, seq_length, gru_hidden_size)

        # Nehme nur letzten Zeitpunkt Output
        last_output = gru_out[:, -1, :]  # Shape: (batch_size, gru_hidden_size)

        # classification layer
        out = self.fc(last_output)  # Shape: (batch_size, num_classes)

        return out
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

data_loader = torch.utils.data.DataLoader(
    dataset_tr,
    batch_size=1,
    shuffle=True,
)

data_loader_test = torch.utils.data.DataLoader(
    dataset_tes,
    batch_size=1,
    shuffle=False,
)
##
print()
# in = data_loader.x

model = TemporalGCN(input_size, hidden_size, num_layers, num_classes)
##

# Vorhersage
y = model(data.x, data.y)
print(y.shape)  # Erwartet: (32, 45)


optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
         out = model(x, adjacency_matrix)  # Perform a single forward pass.
         loss = criterion(out, data.y.view(-1))  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.

def test(loader):
     model.eval()

     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(x, adjacency_matrix)
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.


for epoch in range(1, 100):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)

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

