import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from mkl_random.mklrand import geometric
from torch.utils.data import DataLoader, random_split
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GraphConv
from torch.nn import Linear
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import Data, Batch

from Mydataset import SkeletonDataset

num_joints = 21

transform = NormalizeFeatures()  # Normalisiert die Node-Features auf einen Bereich von [0, 1]
#  File "/home/boris.grillborzer/miniconda3/envs/gggten/lib/python3.10/site-packages/torch_geometric/transforms/normalize_features.py", line 24, in forward
#     for store in data.stores:
# AttributeError: 'Tensor' object has no attribute 'stores'
batch_size = 32
# Dataset und DataLoader
path = osp.normpath(osp.join(osp.dirname(__file__), "First-PersonHandActionBenchmarkF-PHAB"))
dataset = SkeletonDataset(path, transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
node_features, edge_index, y = train_dataset[0]
node_featurestest, edge_indextest, ytest = test_dataset[0]

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=Batch.from_data_list)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=Batch.from_data_list)

data = Data(x=node_features, edge_index=edge_index, y=y)
data.y = data.y.clamp(0, num_joints - 1)
print()

datatest = Data(x=node_featurestest, edge_index=edge_indextest, y=ytest)
datatest.y = data.y.clamp(0, num_joints - 1)

# GCN-Modell
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(3, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 3)

    def forward(self, x, edge_index, batch):

        # 1. Obtain node embeddings

        x = self.conv1(x, edge_index)

        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x

# Modell initialisieren
model = GCN(hidden_channels=64)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
         out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
         loss = criterion(out, data.y.view(-1))  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.

def test(loader):
     model.eval()

     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch)
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.


for epoch in range(1, 171):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

def predict(model, image):
    model.eval()
    with torch.no_grad():
        image = transform(image).unsqueeze(0).cuda()  # Batch-Dimension hinzufügen
        outputs = model(image)
        return outputs.cpu().numpy().flatten()  # Gibt die XYZ-Koordinaten der Keypoints zurück

