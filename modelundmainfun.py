import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import load_all_skeletons
from Mydataset import SkeletonDataset

# Beispiel: 21 Gelenke mit 3 Features (XYZ)
num_joints = 21

node_features, _ = dataset

edge_index = torch.tensor([         # Adjazenzmatrix für 21 Handgelenke (Indexierung korrigiert)
    (0, 1), (1, 2), (2, 3), (3, 4),  # Daumen
    (0, 5), (5, 6), (6, 7), (7, 8),  # Zeigefinger
    (0, 9), (9, 10), (10, 11), (11, 12),  # Mittelfinger
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ringfinger
    (0, 17), (17, 18), (18, 19), (19, 20)   # Kleiner Finger
], dtype=torch.long)

# Transponieren für die Form [2, num_edges]
edge_index = edge_index.t()

# Graph-Datenobjekt
data = Data(x=node_features, edge_index=edge_index)

# GCN-Modell
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# Modell initialisieren
model = GCN(input_dim=3, hidden_dim=32, output_dim=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training-Loop (Dummy-Labels für Pose)
labels = torch.rand(num_joints, 3)  # Zielpose
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.mse_loss(out, labels)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
# node_features = # XYZ-Koordinaten    tensor(21, 3)     tensor([[x, y, z], [x, y, z] .. 21 mal [x, y, z]])
# also wird nur ein Frame ans Model uebergeben!!!
#__________________________________________________mit DataLoader__________________________:_____________________
# DataLoader erstellen
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training mit DataLoader
for epoch in range(num_epochs):
    for data, label in dataloader:
        # Daten und Labels an Modell übergeben
        outputs = model(data)
        loss = loss_fn(outputs, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()