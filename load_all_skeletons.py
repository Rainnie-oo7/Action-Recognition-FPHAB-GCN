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
    print()
    return skeleton_paths, actions

#hier muss aktionen als labels
#wurden die jetzt 0, xyz 1 xyz und so weiter, wenn ja ändern
#targets nach action anpass
#hall#

# [(0, [[x, y, z], [x, y, z], ..]]),     (1, [[x, y, z], [x, y, z], ..]]),    (2, [[x, y, z], [x, y, z], ..]])]
def load_data_from_skeleton_path(paths_dict, labels):
    coordinatesdata = []

    for action, paths in paths_dict.items():
        # Holen des label_value für die aktuelle Aktion aus dem Dictionary
        label_value = labels.get(action, None)

        if label_value is not None:
            # Liste zum Speichern der Skeleton-Daten für diese Aktion
            skeleton_data = []

            for path in paths:
                try:
                    # Lade die Skelettdaten aus der Datei
                    skeleton_onetxt = np.loadtxt(path)
                    # Entferne die erste Spalte
                    skeleton_onetxt = skeleton_onetxt[:, 1:]
                    action_data = [row.reshape(-1, 3) for row in skeleton_onetxt]

                    skeleton_data.append(action_data)
                except Exception as e:
                    print(f"Fehler beim Laden von {path}: {e}")
            skmat = np.asarray(skeleton_data[0])
            # Füge die Skelettdaten als Tupel mit dem label_value hinzu
            for coordinates in skmat:
                coordinatesdata.append((label_value, coordinates))  # Liste der Koordinaten hinzufügen

    return coordinatesdata

def load_coos_from_coordinatesdata(coordinatesdata): #This function extist because I am too dumb to extract coordinates to whole 4551 length from length of coordinatesdata 4551 label-coords tupel
    coords = [twentyone[1] for twentyone in coordinatesdata]
    return coords

def load_labels_from_coordinatesdata1(coordinatesdata):
    labelsss = [twentyone[0] for twentyone in coordinatesdata for _ in range(21)]  # Labels extrahieren
    return labelsss

def load_labels_from_coordinatesdata(labelsss): #This function extist because I am too dumb to extract targetinglabels to whole 4551 length from length of coordinatesdata 4551 label-coords tupel

    # Labels in 21er-Gruppen unterteilen
    labelsss_packed = [np.array(labelsss[i:i + 21]) for i in range(0, len(labelsss), 21)]

    return labelsss_packed

# [(1, [x1, y1, z1]), (1, [x2, y2, z2]), (1, [xn, yn, zn]),     (2, [x1, y1, z1]), (2, [x2, y2, z2]), (2, [xn, yn, zn]),    ...,    (m, [x1, y1, z1]), (m, [x2, y2, z2]), (m, [xn, yn, zn])]
# def load_data_from_skeleton_path(paths_dict, labels):
#     action_list_with_labels = []
#
#     for action, paths in paths_dict.items():
#         # Holen des label_value für die aktuelle Aktion
#         label_value = labels.get(action, None)
#
#         if label_value is not None:
#             for path in paths:
#                 try:
#                     # Lade die Skelettdaten aus der Datei
#                     skeleton_onetxt = np.loadtxt(path)
#                     # Entferne die erste Spalte
#                     skeleton_onetxt = skeleton_onetxt[:, 1:]
#                     # Jede Zeile repräsentiert eine Sammlung von Punkten (z. B. joints)
#                     for row in skeleton_onetxt:
#                         # Teile jede Zeile in Gruppen von [x, y, z]
#                         coordinates = row.reshape(-1, 3)
#                         for coord in coordinates:
#                             action_list_with_labels.append((label_value, coord.tolist()))
#                 except Exception as e:
#                     print(f"Fehler beim Laden von {path}: {e}")
#
#     return action_list_with_labels

#dict aneinadnergekettte lsiten brauche ja nur eine Zeile immer hmm
# def load_data_from_skeleton_path(paths_dict, labels):
#     action_matrices = {}  # Dictionary zur Speicherung von Matrizen pro Label
#
#     for action, paths in paths_dict.items():
#         # Holen des label_value für die aktuelle Aktion aus dem Dictionary
#         label_value = labels.get(action, None)
#
#         if label_value is not None:
#             # Liste zum Speichern aller Koordinaten für diese Aktion
#             skeleton_data = []
#
#             for path in paths:
#                 try:
#                     # Lade die Skelettdaten aus der Datei
#                     skeleton_onetxt = np.loadtxt(path)
#                     # Entferne die erste Spalte (falls sie nicht benötigt wird)
#                     skeleton_onetxt = skeleton_onetxt[:, 1:]
#                     # Konvertiere in eine Liste von [x, y, z] für jeden Frame
#                     action_data = [row.reshape(-1, 3) for row in skeleton_onetxt]
#
#                     skeleton_data.extend(action_data)  # Füge alle Frames hinzu
#                 except Exception as e:
#                     print(f"Fehler beim Laden von {path}: {e}")
#
#             # Erstelle eine NumPy-Matrix aus den gesammelten Koordinaten
#             if skeleton_data:
#                 action_matrix = np.vstack(skeleton_data)  # Form [num_frames, 3]
#                 action_matrices[label_value] = action_matrix  # Speichere Matrix im Dictionary
#
#     return action_matrices

