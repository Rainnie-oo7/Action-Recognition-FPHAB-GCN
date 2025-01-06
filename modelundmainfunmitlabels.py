import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
import load_all_skeletons
from Mydataset import SkeletonDataset

class KeypointDetectionModel(nn.Module):
    def __init__(self, num_keypoints, num_classes):
        super(KeypointDetectionModel, self).__init__()

        # Convolutional Layer: Bildmerkmale extrahieren
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Fully Connected Layer: Umwandlung der Merkmale in Keypoint-Koordinaten und Labels
        self.fc1 = nn.Linear(128 * 64 * 64, 1024)  # Angenommen, das Bild wird auf 64x64 reduziert
        self.fc2 = nn.Linear(1024, num_keypoints * 3)  # Jede Keypoint hat 3 Koordinaten (X, Y, Z)
        self.fc3 = nn.Linear(1024, num_keypoints * num_classes)  # Label-Prognose für jedes Keypoint

        self.relu = nn.ReLU()

    def forward(self, x):
        # CNN-Forward-Pass
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        # Flatten und Weitergabe durch Fully Connected Layer
        x = x.view(x.size(0), -1)  # Flach machen für den FC-Layer
        x = self.relu(self.fc1(x))
        
        # XYZ-Koordinaten der Keypoints vorhersagen
        keypoints = self.fc2(x)

        # Labels für die Keypoints vorhersagen
        labels = self.fc3(x)
        
        return keypoints, labels

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
def train(model, train_loader, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, keypoints, labels in train_loader:
            # Bilder, Keypoints und Labels auf das richtige Device verschieben (z.B. CUDA, wenn verfügbar)
            images = images.cuda()
            keypoints = keypoints.cuda()
            labels = labels.cuda()

            # Optimierungsschritt
            optimizer.zero_grad()

            # Vorwärtsdurchlauf
            pred_keypoints, pred_labels = model(images)

            # Verlust berechnen
            loss = combined_loss(pred_keypoints, pred_labels, keypoints, labels, num_keypoints=17, num_classes=10)
            loss.backward()

            # Optimierungsschritt
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

################################################################
# Hyperparameter
image_paths = [...]  # Liste der Bildpfade
keypoint_annotations = [...]  # Liste der XYZ-Koordinaten (z.B. für 17 Keypoints = 17 * 3 = 51 Werte)
labels = [...]  # Labels für jedes Keypoint (z.B. Klassenzugehörigkeit)

# Bildtransformation (z.B. Normalisierung)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dataset und DataLoader
dataset = SkeletonDataset(image_paths, keypoint_annotations, labels, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Modell initialisieren
num_keypoints = 17  # Anzahl der Keypoints, z.B. 17 für den menschlichen Körper
num_classes = 10  # Anzahl der möglichen Klassen für jedes Keypoint
model = KeypointDetectionModel(num_keypoints, num_classes).cuda()

# Optimizer und Verlustfunktion
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training starten
train(model, train_loader, optimizer, num_epochs=10)

#############################Vorhersage###################################
def predict(model, image):
    model.eval()
    with torch.no_grad():
        image = transform(image).unsqueeze(0).cuda()  # Batch-Dimension hinzufügen
        outputs = model(image)
        return outputs.cpu().numpy().flatten()  # Gibt die XYZ-Koordinaten der Keypoints zurück

