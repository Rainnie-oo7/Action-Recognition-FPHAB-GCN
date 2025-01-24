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
from torch_geometric.loader import DataLoader
from get_data_to_list import  get_data_to_list
from torch_geometric.transforms import NormalizeFeatures

# transform = NormalizeFeatures()  # Normalisiert die Node-Features auf einen Bereich von [0, 1]

path = osp.normpath(osp.join(osp.dirname(__file__), "First-PersonHandActionBenchmarkF-PHAB"))

train_data = get_data_to_list(path)
print()
# #Transformiere die Daten (z. B. Normalisierung)
# if transform:
#     train_data = transform(train_data)

num_joints = 21

#  File "/home/boris.grillborzer/miniconda3/envs/gggten/lib/python3.10/site-packages/torch_geometric/transforms/normalize_features.py", line 24, in forward
#     for store in data.stores:
# AttributeError: 'Tensor' object has no attribute 'stores'
batch_size = 32
# Dataset und DataLoader

train_size = int(0.8 * len(train_data))
test_size = len(train_data) - train_size

train_dataset, test_dataset = random_split(train_data, [train_size, test_size])
data = train_dataset
datatest = test_dataset

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# GCN-Modell
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(3, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 45)

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
model = GCN(hidden_channels=128)

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


for epoch in range(1, 10000):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)

    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    # Adapt the learning rate based on the epoch    #Weight Decay?
    if epoch == 50:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.1 * param_group['lr']

#Epoch: 322, Train Acc: 0.5431, Test Acc: 0.5379
#Epoch: 322, Train Acc: 0.7407, Test Acc: 0.7289    second run with weight decay
#Epoch: 983, Train Acc: 0.8275, Test Acc: 0.8189        v
#Epoch:1381, Train Acc: 0.8429, Test Acc: 0.8321
#Epoch:1860, Train Acc: 0.8651, Test Acc: 0.8617
#Epoch: 2064, Train Acc: 0.8854, Test Acc: 0.8749
#Epoch: 2502, Train Acc: 0.8813, Test Acc: 0.8836

# Epoch: 5657, Train Acc: 0.9118, Test Acc: 0.9001

#Epoch 9714, Train Acc: 0.9497 E 8584, Test Acc: 0.9308 Max.

def predict(model, image):
    model.eval()
    with torch.no_grad():
        image = transform(image).unsqueeze(0).cuda()  # Batch-Dimension hinzufügen
        outputs = model(image)
        return outputs.cpu().numpy().flatten()  # Gibt die XYZ-Koordinaten der Keypoints zurück

