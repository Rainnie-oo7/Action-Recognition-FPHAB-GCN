import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.nn import Linear
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
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

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
model = GCN(input_dim=3, hidden_dim=32, output_dim=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# siehe 2_PoseEstimation.libre office
def combined_loss(pred_keypoints, pred_labels, true_keypoints, true_labels, num_keypoints, num_classes):
    # MSE Loss für Keypoints (XYZ-Koordinaten)
    keypoint_loss = nn.MSELoss()(pred_keypoints, true_keypoints)
    
    # Cross-Entropy Loss für die Labels (Klassifikation)
    label_loss = 0
    for i in range(num_keypoints):
        label_loss += nn.CrossEntropyLoss()(pred_labels[:, i*num_classes:(i+1)*num_classes], true_labels[:, i])

    # Gesamter Verlust: Kombination von Keypoint- und Label-Verlust
    total_loss = keypoint_loss + label_loss
    return total_loss
#______________________________________________________

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
        
#__________________________________________________________
def train(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, keypoints in train_loader:
            # Bilder und Keypoints auf das richtige Device verschieben (z.B. CUDA, wenn verfügbar)
            images = images.cuda()
            keypoints = keypoints.cuda()

            # Optimierungsschritt
            optimizer.zero_grad()

            # Vorwärtsdurchlauf
            outputs = model(images)

            # Verlust berechnen
            loss = criterion(outputs, keypoints)
            loss.backward()

            # Optimierungsschritt
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

################################################################
# Hyperparameter
image_paths = [...]  # Liste der Bildpfade
keypoint_annotations = [...]  # Liste der XYZ-Koordinaten (z.B. für 17 Keypoints = 17 * 3 = 51 Werte)

# Bildtransformation (z.B. Normalisierung)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dataset und DataLoader
dataset = KeypointDataset(image_paths, keypoint_annotations, transform=transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Modell initialisieren
num_keypoints = 17  # Anzahl der Keypoints, z.B. 17 für den menschlichen Körper
model = KeypointDetectionModel(num_keypoints).cuda()

# Optimizer und Verlustfunktion
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training starten
train(model, train_loader, criterion, optimizer)
#############################Vorhersage###################################
def predict(model, image):
    model.eval()
    with torch.no_grad():
        image = transform(image).unsqueeze(0).cuda()  # Batch-Dimension hinzufügen
        outputs = model(image)
        return outputs.cpu().numpy().flatten()  # Gibt die XYZ-Koordinaten der Keypoints zurück

