import json
import os
import numpy as np

# Skelettdefinition
coco_skeleton_definition = {
    "keypoints": [
        "r_ankle", "r_knee", "r_hip", "l_hip", "l_knee", "l_ankle",
        "pelvis", "thorax", "upper_neck", "head_top", "r_wrist", "r_elbow", "r_shoulder",
        "l_shoulder", "l_elbow", "l_wrist"
    ],
    "skeleton": [
        [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [3, 7], [7, 8], [8, 9], [9, 10],
        [7, 12], [12, 11], [11, 10], [7, 13], [13, 14], [14, 15]
    ]
}

# Funktion zum Generieren von Bewegungssequenzen
def generate_skeleton_sequence(output_dir, num_frames, noise_level=0.1):
    os.makedirs(output_dir, exist_ok=True)

    # Startposition der Keypoints (x, y, z)
    base_keypoints = np.array([
        [0, 0, 0],  # r_ankle
        [0, 5, 0],  # r_knee
        [0, 10, 0], # r_hip
        [5, 10, 0], # l_hip
        [5, 5, 0],  # l_knee
        [5, 0, 0],  # l_ankle
        [2.5, 10, 0], # pelvis
        [2.5, 15, 0], # thorax
        [2.5, 17, 0], # upper_neck
        [2.5, 19, 0], # head_top
        [0, 15, -5], # r_wrist
        [0, 15, -2.5], # r_elbow
        [0, 15, 0],  # r_shoulder
        [5, 15, 0],  # l_shoulder
        [5, 15, -2.5], # l_elbow
        [5, 15, -5]  # l_wrist
    ])

    for frame in range(num_frames):
        keypoints = base_keypoints.copy()

        # Hinzufügen von Bewegung (z.B. "Hand heben")
        angle = np.radians(10 * frame)  # Winkel für die Bewegung
        keypoints[10, 2] += 5 * np.sin(angle)  # r_wrist
        keypoints[11, 2] += 2.5 * np.sin(angle)  # r_elbow

        # Hinzufügen von zufälligem Rauschen
        noise = np.random.uniform(-noise_level, noise_level, keypoints.shape)
        keypoints += noise

        # Keypoints in COCO-Format (flaches Array)
        keypoints_flat = keypoints.flatten().tolist()

        # JSON-Daten vorbereiten
        json_data = {
            "annotations": [{
                "keypoints": keypoints_flat
            }],
            "categories": [{
                "skeleton": coco_skeleton_definition["skeleton"]
            }]
        }

        # JSON-Datei speichern
        output_path = os.path.join(output_dir, f"frame_{frame:03d}.json")
        with open(output_path, "w") as json_file:
            json.dump(json_data, json_file, indent=4)

        print(f"Frame {frame} gespeichert: {output_path}")

# Beispiel: Sequenz generieren
output_directory = "generated_coco_skeletons"
generate_skeleton_sequence(output_directory, num_frames=1, noise_level=0.1)
