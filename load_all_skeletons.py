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
            for subjectdir in sorted(os.listdir(dir_path)):
                subject_path = os.path.join(dir_path, subjectdir)
                if os.path.isdir(subject_path):
                    for actiondir in sorted(os.listdir(subject_path)):
                        action_path = os.path.join(subject_path, actiondir)
                        if os.path.isdir(action_path):
                        # Group skeleton paths by action # Init Liste für die Aktion, wenn sie noch nicht vorhanden,
                            if actiondir not in skeleton_paths:
                                skeleton_paths[actiondir] = []
                                actions.append(actiondir)
                            for participiantdir in sorted(os.listdir(action_path)):
                                participiant_path = os.path.join(action_path, participiantdir)
                                if os.path.isdir(subject_path):
                                    skeleton_txtpath = os.path.join(participiant_path, 'skeleton.txt')
                                    # Appende den Pfad in die Liste der entsprechenden Aktion
                                    skeleton_paths[actiondir].append(skeleton_txtpath)

    return skeleton_paths, actions

# [(0, [[x, y, z], [x, y, z], ..]]),     (1, [[x, y, z], [x, y, z], ..]]),    (2, [[x, y, z], [x, y, z], ..]])]
# def load_data_from_skeleton_path(paths_dict, labels):
#     action_list_with_labels = []
#
#     for action, paths in paths_dict.items():
#         # Holen des label_value für die aktuelle Aktion aus dem Dictionary
#         label_value = labels.get(action, None)
#
#         if label_value is not None:
#             # Liste zum Speichern der Skeleton-Daten für diese Aktion
#             skeleton_data = []
#
#             for path in paths:
#                 try:
#                     # Lade die Skelettdaten aus der Datei
#                     skeleton_onetxt = np.loadtxt(path)
#                     # Entferne die erste Spalte
#                     skeleton_onetxt = skeleton_onetxt[:, 1:]
#                     action_data = [row.reshape(-1, 3) for row in skeleton_onetxt]
#
#                     skeleton_data.append(action_data)
#                 except Exception as e:
#                     print(f"Fehler beim Laden von {path}: {e}")
#
#             # Füge die Skelettdaten als Tupel mit dem label_value hinzu
#             for coordinates in skeleton_data:
#                 for coord in coordinates:
#                     action_list_with_labels.append((label_value, coord.tolist()))  # Liste der Koordinaten hinzufügen
#
#     return action_list_with_labels

def load_data_from_skeleton_path(paths_dict, labels):
    action_list_with_labels = []

    for action, paths in paths_dict.items():
        # Holen des label_value für die aktuelle Aktion
        label_value = labels.get(action, None)

        if label_value is not None:
            for path in paths:
                try:
                    # Lade die Skelettdaten aus der Datei
                    skeleton_onetxt = np.loadtxt(path)
                    # Entferne die erste Spalte
                    skeleton_onetxt = skeleton_onetxt[:, 1:]
                    # Jede Zeile repräsentiert eine Sammlung von Punkten (z. B. joints)
                    for row in skeleton_onetxt:
                        # Teile jede Zeile in Gruppen von [x, y, z]
                        coordinates = row.reshape(-1, 3)
                        for coord in coordinates:
                            action_list_with_labels.append((label_value, coord.tolist()))
                except Exception as e:
                    print(f"Fehler beim Laden von {path}: {e}")

    return action_list_with_labels

