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
from get_data_to_list import  get_data_to_list

# transform = NormalizeFeatures()  # Normalisiert die Node-Features auf einen Bereich von [0, 1]

path = osp.normpath(osp.join(osp.dirname(__file__), "First-PersonHandActionBenchmarkF-PHAB"))

train_data = get_data_to_list(path)
#4551 Liste von Objekten á:  x, edgeindex, y

# #Transformiere die Daten (z. B. Normalisierung)
# if transform:
#     train_data = transform(train_data)

batch_size = 1

train_size = int(0.8 * len(train_data))
test_size = len(train_data) - train_size

train_dataset, test_dataset = random_split(train_data, [train_size, test_size])
data = train_dataset
datatest = test_dataset

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class TemporalGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TemporalGraphConv, self).__init__()
        self.gcn = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        # Spatial Graph Convolution
        x = self.gcn(x, edge_index)
        x = F.relu(x)
        return x


class TemporalGraphNNWithMemory(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_layers=2):
        super(TemporalGraphNNWithMemory, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.append(TemporalGraphConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.layers.append(TemporalGraphConv(hidden_channels, hidden_channels))

        self.lstm = nn.LSTM(hidden_channels, hidden_channels, batch_first=True)

        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)

        total_elements = x.numel()  #total_elements des Batches = 1323
        time_steps = 10

        num_features = x.size(1)  # 63

        batch_size = x.size(0)  # 21

        # assert total_elements == batch_size * time_steps * num_features, \
        #     f"Shape mismatch: {total_elements} elements vs {batch_size * time_steps * num_features}"

        # Reshape der Tensorform zu [Batch, Time, Features]
        x = x.view(batch_size, time_steps, 3)

        # Gedächtnis (LSTM)
        x, _ = self.lstm(x)

        # Global Pooling und Klassifikation
        x = torch.mean(x, dim=1)  # Pooling über die Zeitachse
        return self.classifier(x)


# Modell initialisieren
in_channels = train_data[0].x.shape[1]  # Nimm die Anzahl der Merkmale der ersten Instanz

model = TemporalGraphNNWithMemory(in_channels=in_channels, hidden_channels=63, num_classes=45)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
         out = model(data.x, data.edge_index)  # Perform a single forward pass.
         loss = criterion(out, data.y.view(-1))  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.

def test(loader):
     model.eval()

     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index)
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

#Epoch: 322, Train Acc: 0.5431, Test Acc: 0.5379
#Epoch: 322, Train Acc: 0.7407, Test Acc: 0.7289    second run with weight decay
#Epoch: 983, Train Acc: 0.8275, Test Acc: 0.8189        v
#Epoch:1381, Train Acc: 0.8429, Test Acc: 0.8321
#Epoch:1860, Train Acc: 0.8651, Test Acc: 0.8617
#Epoch: 2064, Train Acc: 0.8854, Test Acc: 0.8749
#Epoch: 2502, Train Acc: 0.8813, Test Acc: 0.8836

# Epoch: 5657, Train Acc: 0.9118, Test Acc: 0.9001

#Epoch 9714, Train Acc: 0.9497 E 8584, Test Acc: 0.9308 Max.

# def predict(model, image):
#     model.eval()
#     with torch.no_grad():
#         image = transform(image).unsqueeze(0).cuda()  # Batch-Dimension hinzufügen
#         outputs = model(image)
#         return outputs.cpu().numpy().flatten()  # Gibt die XYZ-Koordinaten der Keypoints zurück

