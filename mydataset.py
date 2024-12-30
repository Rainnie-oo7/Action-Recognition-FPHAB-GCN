import os
import numpy as np


def load_all_skeletons(data_root):
    skeleton_data = []

    # Iteriere durch alle Subjects
    for subjectdir in os.listdir(data_root):
        subject_path = os.path.join(data_root, subjectdir)
        if os.path.isdir(subject_path):
            # Iteriere durch alle Aktionen
            for actiondir in os.listdir(subject_path):
                action_path = os.path.join(subject_path, actiondir)
                if os.path.isdir(action_path):
                    # Iteriere durch alle Sequenzen
                    for sequencedir in os.listdir(action_path):
                        sequence_path = os.path.join(action_path, sequencedir)
                        skeleton_file = os.path.join(sequence_path, 'skeleton.txt')
                        if os.path.isfile(skeleton_file):
                            skeleton = np.loadtxt(skeleton_file)

                            # label = f"{subjectdir}/{actiondir}/{sequencedir}"

                            # skeleton_data.append({
                            #     'label': label,
                            #     'skeleton': skeleton      # Da wir erst mit Pose Estimierung anfangen lasssen wir Action Recognition aussenvor.
                            # })
                            skeleton_data.append({
                                'skeleton': skeleton
                            })


    return skeleton_data


data_root = r"C:\Users\Boris Grillborzer\PycharmProjects\PoseEstimation\First-PersonHandActionBenchmarkF-PHAB\Hand_pose_annotation_v1"
skeleton_data = load_all_skeletons(data_root)

# Ausgabe: Anzahl der geladenen Datensätze
print(f"Es wurden {len(skeleton_data)} Skelettdatensätze geladen.") #1178

# Beispiel: Anzeige eines Labels und der ersten paar Skelettwerte
# print(skeleton_data[0]['label'])
# print(skeleton_data[0]['skeleton'][:5])
c=skeleton_data[0]['skeleton'][:,1:]
print(c)
print("ABC")