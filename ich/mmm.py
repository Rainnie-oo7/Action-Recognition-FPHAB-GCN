import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import MyDataset

# Beispiel-Daten: Graph-Datenstruktur mit 16 Punkten
# Dummy-Daten: Jeder Knoten hat 2 Features (x- und y-Koordinaten)

def create_graph_data():
    num_nodes = 16  # 16 Punkte im MPII-Dataset

    # Beispielhafte Rohdaten vorbereiten\n
    raw_data = [
        {'keypoints': [[0.1, 0.2], [0.3, 0.4], ...],
         'labels': [[0.15, 0.25], ...]},
        {'keypoints': [[0.5, 0.6], [0.7, 0.8], ...],
         'labels': [[0.55, 0.65], ...]}]  # Weitere Beispiele hinzufügen
    # Dataset erstellen
    pose_dataset = MyDataset(raw_data)
    # Graph-Daten erstellen
    data_list = [create_graph_data(pose_dataset, idx) for idx in range(len(pose_dataset))]

    ### Dynamisches Laden mit DataLoader
    #Falls Sie die Daten direkt in einem DataLoader verwenden möchten, ist das auch möglich:


    dataloader = DataLoader(pose_dataset, batch_size=32, shuffle=True)

    for batch_idx, (x, y) in enumerate(dataloader):# Daten verarbeiten (z. B. erstellen von Graph-Daten mit create_graph_data())\n    batch_graphs = [create_graph_data(pose_dataset, idx) for idx in range(batch_idx, batch_idx + len(x))]\n    # Weiterverarbeitung der Graph-Daten\n```

        x = x

    0 r ankle, 1 r knee, 2 r hip, 3 l hip, 4 l knee, 5 l ankle, 6 pelvis, 7 thorax,
     8 upper neck, 9 head top, 10 r wrist, 11 r elbow, 12 r shoulder, 13 l shoulder,
     14 l elbow, 15 l wrist


    edges = [   # Adjazenzinformationen: Edges (Kanten des Graphen)
        (15, 14), (14, 13), (10, 11), (11, 12),
        (12, 7), (13, 7), (7, 8), (8, 9),
        (2, 6), (3, 6), (2, 12), (3, 13),
        (0, 1), (1, 2), (3, 4), (4, 5)]


    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Labels: Zufällige Zielkoordinaten
    y = torch.rand((num_nodes, 2))

    return Data(x=x, edge_index=edge_index, y=y)

# Dataset erstellen
data_list = [create_graph_data() for _ in range(1000)]
dataset = DataLoader(data_list, batch_size=32, shuffle=True)

# GNN-Modell (Graph Convolutional Network)
class PoseGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PoseGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = self.fc(x)  # Ausgabe: Koordinaten der Punkte
        return x

# Modell, Optimizer, Loss
model = PoseGNN(input_dim=2, hidden_dim=64, output_dim=2)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Training
for epoch in range(50):
    model.train()
    total_loss = 0

    for batch in dataset:
        optimizer.zero_grad()
        out = model(batch)  # Vorhersage der Punktkoordinaten
        loss = criterion(out, batch.y)  # MSE-Loss zwischen Vorhersage und Ground Truth
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

print("Training abgeschlossen!")
