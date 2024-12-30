import json
import os
import random
import numpy as np

# Einstellungen
output_dir = "coco_skeletons"
os.makedirs(output_dir, exist_ok=True)

# Parameter der Sequenz
num_frames = 1  # Anzahl der Frames
keypoints_template = [
    [300, 750, 2],  # Kopf (x, y, Sichtbarkeit)
    [300, 700, 2],  # Hals
    [250, 600, 2],  # Linke Schulter
    [350, 600, 2],  # Rechte Schulter
    [250, 500, 2],  # Linker Ellbogen
    [350, 500, 2],  # Rechter Ellbogen
    [250, 400, 2],  # Linke Hand
    [350, 400, 2],  # Rechte Hand
]

# Bewegung simulieren: Hand heben (rechte Hand nach oben bewegen)
def generate_keypoints_sequence(template, frames):
    keypoints_sequence = []
    for frame in range(frames):
        keypoints = []
        for i, (x, y, visibility) in enumerate(template):
            # Rechte Hand nach oben bewegen
            if i == 7:  # Index der rechten Hand
                y -= frame * 20  # Hand heben pro Frame

            # Zufälliges Rauschen hinzufügen
            x += random.uniform(-5, 5)
            y += random.uniform(-5, 5)

            # Hinzufügen der Koordinate
            keypoints.extend([x, y, visibility])

        keypoints_sequence.append(keypoints)
    return keypoints_sequence

# Generiere Sequenz
keypoints_sequence = generate_keypoints_sequence(keypoints_template, num_frames)

# JSON-Dateien erstellen
def save_to_coco_json(keypoints, frame_index, output_directory):
    json_data = {
        "info": {
            "year": 2024,
            "version": "1.0",
            "description": "Bewegung: Hand heben",
            "contributor": "Automatisch generiert",
            "date_created": "2024-12-24"
        },
        "images": [
            {
                "id": 1,
                "width": 640,
                "height": 480,
                "file_name": f"frame_{frame_index:03d}.jpg"
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "keypoints": keypoints,
                "num_keypoints": len(keypoints) // 3
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": "person",
                "supercategory": "human",
                "keypoints": [
                    "head", "neck", "left_shoulder", "right_shoulder",
                    "left_elbow", "right_elbow", "left_hand", "right_hand"
                ],
                "skeleton": [
                    [1, 2], [2, 3], [2, 4], [3, 5], [4, 6], [5, 7], [6, 8]
                ]
            }
        ]
    }

    # Speichern der Datei
    output_path = os.path.join(output_directory, f"frame_{frame_index:03d}.json")
    with open(output_path, "w") as json_file:
        json.dump(json_data, json_file, indent=4)

# Speichere jede Position als JSON-Datei
for i, keypoints in enumerate(keypoints_sequence):
    save_to_coco_json(keypoints, i + 1, output_dir)

print(f"{num_frames} JSON-Dateien wurden im Verzeichnis '{output_dir}' gespeichert.")
