import os
import numpy as np
import os.path as osp
import torch

def load_all_skeletons(data_root):
    actions = []
    skeleton_paths = {}
    target_dir = os.path.join(data_root, 'Hand_pose_annotation_v1')
    # Iteriere durch alle Subjects
    for dir in os.listdir(data_root):
        dir_path = os.path.join(data_root, dir)
        if dir_path == target_dir:
            # Iteriere durch alle Aktionen
            for subjectdir in os.listdir(dir_path):
                subject_path = os.path.join(dir_path, subjectdir)
                if os.path.isdir(subject_path):
                    for actiondir in os.listdir(subject_path):
                        action_path = os.path.join(subject_path, actiondir)
                        if os.path.isdir(action_path):
                        # Group skeleton paths by action # Init Liste für die Aktion, wenn sie noch nicht vorhanden,
                            if actiondir not in skeleton_paths:
                                skeleton_paths[actiondir] = []
                                actions.append(actiondir)
                            for participiantdir in os.listdir(action_path):
                                participiant_path = os.path.join(action_path, participiantdir)
                                if os.path.isdir(subject_path):
                                    skeleton_txtpath = os.path.join(participiant_path, 'skeleton.txt')
                                    # Appende den Pfad in die Liste der entsprechenden Aktion
                                    skeleton_paths[actiondir].append(skeleton_txtpath)

    return skeleton_paths, actions

#Lade aus Dictionary-verpackte-Action-gruppierte Pfade
def load_data_from_skeleton_path(paths_dict):
    skeleton_data = {}
    for action, paths in paths_dict.items():
        if action == 'charge_cell_phone':
            # Liste zum Speichern der Skeleton-Daten für diese Aktion
            grouped_data_list = []
            action_skeletons = []

            for path in paths:

                try:
                    # Lade die Skelettdaten aus der Datei
                    skeleton_onetxt = np.loadtxt(path)
                    # Entferne die erste Spalte
                    skeleton_onetxt = skeleton_onetxt[:, 1:]
                    grouped_data = [row.reshape(-1, 3) for row in skeleton_onetxt]
                    #ndarray 79, 63 --> 79, 21, 3
                    # node_features = torch.tensor([[x1, y1, z1],
                    # [x2, y2, z2],
                    # [x3, y3, z3],
                    # ...])

                    action_skeletons.append(skeleton_onetxt)
                    grouped_data_list.append(grouped_data)
                except Exception as e:
                    print(f"Fehler beim Laden von {path}: {e}")

            # Speichere die geladenen Skeleton-Daten unter dem Aktionsnamen
            skeleton_data[action] = action_skeletons

    return skeleton_data, grouped_data_list

# def mk_triplets(skeleton_onetxt):
#     # Create the list of coordinates by iterating through the numbers in steps of 3
#     coordinates = []
#     for i in range(skeleton_onetxt.shape[0]):    #Rows
#
#         for j in range(0, skeleton_onetxt.shape[1], 62):
#             coordinates.append(skeleton_onetxt[:,  j:j + 3])
#     print()
#     # Convert the list of coordinates to a PyTorch tensor
#     node_features = torch.tensor(coordinates)
#
#     # Output the tensor
#     print(node_features)
# #Fehler beim Laden von  Hand_pose_annotation_v1/Subject_6/charge_cell_phone/3/skeleton.txt: expected sequence of length 3 at dim 1 (got 2)
#
# def bobtest(data):
#     # Beispiel ndarray (79 Zeilen, 63 Spalten)
#
#
#     # Prüfen, ob die Anzahl der Spalten durch 3 teilbar ist
#     if data.shape[1] % 3 != 0:
#         raise ValueError("Die Anzahl der Spalten ist nicht durch 3 teilbar.")
#
#     # Jede Zeile in Gruppen von 3 spalten
#     grouped_data = [row.reshape(-1, 3) for row in data]
#
#     # Gruppierte Daten anzeigen
#     for i, row in enumerate(grouped_data[:5]):  # Zeige die ersten 5 Zeilen als Beispiel
#         print(f"Zeile {i + 1}:")
#         print(row)
#
#     return grouped_data
