import json
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Funktion zum Laden einer JSON-Datei und Extrahieren der Keypoints und Skeleton
def load_coco_json(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    keypoints = data['annotations'][0]['keypoints']
    skeleton = data['categories'][0]['skeleton']
    return keypoints, skeleton

# COCO-Skelett anzeigen und interaktiv manipulieren
def visualize_coco_skeleton(keypoints, skeleton):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Keypoints extrahieren
    xs = [keypoints[i * 3] for i in range(len(keypoints) // 3)]
    ys = [keypoints[i * 3 + 1] for i in range(len(keypoints) // 3)]
    zs = [keypoints[i * 3 + 2] * 10 for i in range(len(keypoints) // 3)]  # Sichtbarkeit als Tiefe

    # Punkte plotten
    ax.scatter(xs, ys, zs, c='r', marker='o')

    # Verbindungen plotten
    for start, end in skeleton:
        ax.plot(
            [xs[start - 1], xs[end - 1]],
            [ys[start - 1], ys[end - 1]],
            [zs[start - 1], zs[end - 1]],
            c='b'
        )

    # Ansicht von oben setzen
    ax.view_init(elev=90, azim=-90)  # Elevation 90 Grad, Azimut -90 Grad

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

# Mehrere JSON-Dateien nacheinander anzeigen
def visualize_sequence(directory, start_frame, end_frame):
    for b in range(start_frame, end_frame + 1):
        json_path = os.path.join(directory, f"frame_{b:03d}.json")
        if os.path.exists(json_path):
            keypoints, skeleton = load_coco_json(json_path)
            visualize_coco_skeleton(keypoints, skeleton)
        else:
            print(f"Die JSON-Datei {json_path} wurde nicht gefunden.")

# Beispiel: Mehrere JSON-Dateien anzeigen
sequence_dir = "coco_skeletons"
# sequence_dir = "generated_coco_skeletons"
visualize_sequence(sequence_dir, 1, 10)
