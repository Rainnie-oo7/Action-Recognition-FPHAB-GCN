import os
import numpy as np
import os.path as osp



def load_all_skeletons(data_root):
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

def load_skeleton(txtfile_paths):
    skeleton_list2d = []
    frames_nrs = []
    counter = 0
    # Erste 3 Columns unwichtig \\ ignoren.
    line_length = 3 + 25 * 7

    for txtpath in txtfile_paths:

        with open(txtpath) as f:
            for line in f.readlines():
                line = line.strip()
                if "Version" in line:
                    continue
                words = line.split(",")
                frame_nr = int(words[0])    #Erstes Column
                joints2d = {}
                # print(line)
                # print(len(words), line_length)
                if len(words) != line_length:
                    continue

                for idx in range(25):
                    joint_name = words[3 + idx * 7].strip('(')
                    # joint_tracked = words[3 + idx * 7 + 1]

                    # joint_x = float(words[3 + idx * 7 + 2])
                    # joint_y = float(words[3 + idx * 7 + 3])
                    # joint_z = float(words[3 + idx * 7 + 4])

                    joint_dx = float(words[3 + idx * 7 + 5])
                    joint_dy = float(words[3 + idx * 7 + 6].strip(')'))
                    # if joint_x == 0 and joint_y == 0 and joint_z == 0:
                    #     joint_dx = 0
                    #     joint_dy = 0
                    # else:
                    #     joint_dx = float(words[3 + idx * 7 + 5])
                    #     joint_dy = float(words[3 + idx * 7 + 6].strip(')'))
                    # joints5d[joint_name] = [joint_x, joint_y, joint_z, joint_dx, joint_dy]
                    # joints3d[joint_name] = [joint_x, joint_y, joint_z]
                    joints2d[joint_name] = [joint_dx, joint_dy]

                skeleton_list2d.append(joints2d)
                frames_nrs.append(frame_nr)
                counter += 1

    return skeleton_list2d, frames_nrs, counter

