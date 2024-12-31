import os
import numpy as np


def load_all_skeletons(data_root):
    skeleton_paths = {}

    # Iteriere durch alle Subjects
    for subjectdir in os.listdir(data_root):
        subject_path = os.path.join(data_root, subjectdir)
        if os.path.isdir(subject_path):
            # Iteriere durch alle Aktionen
            for actiondir in os.listdir(subject_path):
                action_path = os.path.join(subject_path, actiondir)
                if os.path.isdir(action_path):
                    # Group skeleton paths by action # Init Liste für die Aktion, wenn sie noch nicht vorhanden,
                    if actiondir not in skeleton_paths:
                        skeleton_paths[actiondir] = []

                    # Iteriere durch alle Sequenzen
                    for sequencedir in os.listdir(action_path):
                        sequence_path = os.path.join(action_path, sequencedir)
                        skeleton_file = os.path.join(sequence_path, 'skeleton.txt')
                        if os.path.isfile(skeleton_file):
                            # Appende den Pfad in die Liste der entsprechenden Aktion
                            skeleton_paths[actiondir].append(skeleton_file)


    return skeleton_paths

#Lade aus Dictionary-verpackte-Action-gruppierte Pfade
def load_data_from_skeleton_path(paths_dict):
    skeleton_data = {}

    for action, paths in paths_dict.items():
        # Liste zum Speichern der Skeleton-Daten für diese Aktion
        action_skeletons = []

        for path in paths:
            try:
                # Lade die Skelettdaten aus der Datei
                skeleton_onetxt = np.loadtxt(path)
                # Entferne die erste Spalte
                skeleton_onetxt = skeleton_onetxt[:, 1:]

                action_skeletons.append(skeleton_onetxt)
            except Exception as e:
                print(f"Fehler beim Laden von {path}: {e}")

        # Speichere die geladenen Skeleton-Daten unter dem Aktionsnamen
        skeleton_data[action] = action_skeletons

    return skeleton_data


