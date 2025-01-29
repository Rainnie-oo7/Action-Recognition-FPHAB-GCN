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
from clean_load import  get_data_to_list

path = osp.normpath(osp.join(osp.dirname(__file__), "First-PersonHandActionBenchmarkF-PHAB"))
##
train_data = get_data_to_list(path)
## TODO: = x!!! Muss noch in richtige Shape gebracht werden (batch_size, sequence_length, nodes, features) 4551 Liste von Objekten á:  x, edgeindex, y
##

class GraphConv(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConv, self).__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, adjacency_matrix):
        # x: Tensor of shape (batch_size, nodes, in_features)
        # adjacency_matrix: Tensor of shape (nodes, nodes)
        out = torch.matmul(adjacency_matrix, x)  # Shape: (batch_size, nodes, in_features)
        out = self.fc(out)  # Shape: (batch_size, nodes, out_features)
        return out

class TemporalGCN(nn.Module):
    def __init__(self, in_features, hidden_features, gru_hidden_size, num_classes, num_nodes):
        super(TemporalGCN, self).__init__()
        self.gcn = GraphConv(in_features, hidden_features)
        self.gru = nn.GRU(hidden_features * num_nodes, gru_hidden_size, batch_first=True)
        self.fc = nn.Linear(gru_hidden_size, num_classes)

    def forward(self, x, adjacency_matrix):
        # x: Tensor of shape (batch_size, sequence_length, nodes, features)
        # adjacency_matrix: Tensor of shape (nodes, nodes)
        batch_size, seq_length, num_nodes, in_features = x.size()

        # Apply GCN at each time step
        gcn_out = []
        for t in range(seq_length):
            gcn_out.append(self.gcn(x[:, t], adjacency_matrix))  # Shape: (batch_size, nodes, hidden_features)
        gcn_out = torch.stack(gcn_out, dim=1)  # Shape: (batch_size, seq_length, nodes, hidden_features)

        gru_input = gcn_out.view(batch_size, seq_length, -1)  # Shape: (batch_size, seq_length, nodes * hidden_features)

        gru_out, _ = self.gru(gru_input)  # Shape: (batch_size, seq_length, gru_hidden_size)

        # Nehme nur letzten Zeitpunkt Output
        last_output = gru_out[:, -1, :]  # Shape: (batch_size, gru_hidden_size)

        # classification layer
        out = self.fc(last_output)  # Shape: (batch_size, num_classes)

        return out
##
# Beispiel-Daten
batch_size = 32
sequence_length = 45
num_nodes = 21
in_features = 3
hidden_features = 64
gru_hidden_size = 128
num_classes = 45

x = torch.randn(batch_size, sequence_length, num_nodes, in_features)
adjacency_matrix = torch.tensor([
 [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
 [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]], dtype=torch.float32)  # Shape: (21, 21)
##

train_size = int(0.8 * len(train_data))
test_size = len(train_data) - train_size

train_dataset, test_dataset = random_split(train_data, [train_size, test_size])
data = train_dataset
datatest = test_dataset

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
##

model = TemporalGCN(in_features, hidden_features, gru_hidden_size, num_classes, num_nodes)
##

# Vorhersage
y = model(x, adjacency_matrix)
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

