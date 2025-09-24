import os
import numpy as np
import cv2
import json

# ==========================
# CONFIGURE THESE PATHS
# ==========================
# Root of the VoD dataset (choose radar/lidar and training)
# Camera images from lidar folder
img_dir = r"C:\Users\Budge\OneDrive\Desktop\Schoolwork\Semester7\CSE486\CenterfusionVodTraining\centerfusion-vod-training\data\View_of_Delft_dataset_PUBLIC\view_of_delft_PUBLIC\view_of_delft_PUBLIC\lidar\training\image_2"
label_dir = r"C:\Users\Budge\OneDrive\Desktop\Schoolwork\Semester7\CSE486\CenterfusionVodTraining\centerfusion-vod-training\data\View_of_Delft_dataset_PUBLIC\view_of_delft_PUBLIC\view_of_delft_PUBLIC\lidar\training\label_2"

# Radar data from radar folder
radar_dir = r"C:\Users\Budge\OneDrive\Desktop\Schoolwork\Semester7\CSE486\CenterfusionVodTraining\centerfusion-vod-training\data\View_of_Delft_dataset_PUBLIC\view_of_delft_PUBLIC\view_of_delft_PUBLIC\radar\training\velodyne"
calib_dir = r"C:\Users\Budge\OneDrive\Desktop\Schoolwork\Semester7\CSE486\CenterfusionVodTraining\centerfusion-vod-training\data\View_of_Delft_dataset_PUBLIC\view_of_delft_PUBLIC\view_of_delft_PUBLIC\radar\training\calib"

# Output folder
output_root = r"C:\Users\Budge\OneDrive\Desktop\Schoolwork\Semester7\CSE486\CenterfusionVodTraining\centerfusion-vod-training\data\vod_processed"
image_out_dir = os.path.join(output_root, "images")
radar_out_dir = os.path.join(output_root, "radar")
os.makedirs(image_out_dir, exist_ok=True)
os.makedirs(radar_out_dir, exist_ok=True)

# ==========================
# HELPER FUNCTIONS
# ==========================
def load_kitti_calib(calib_file):
    calib = {}
    with open(calib_file, 'r') as f:
        for line in f:
            if line.startswith("P2:"):
                calib['P2'] = np.array([float(x) for x in line.strip().split()[1:]]).reshape(3, 4)
            elif line.startswith("Tr_velo_to_cam:"):
                calib['Tr_velo_to_cam'] = np.array([float(x) for x in line.strip().split()[1:]]).reshape(3, 4)
    return calib

def load_kitti_label(label_file):
    objects = []
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            cls = parts[0]
            h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
            x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
            yaw = float(parts[14])
            objects.append({"class": cls, "bbox_3d": [x, y, z, l, w, h, yaw]})
    return objects

def lidar_to_camera(box, Tr_velo_to_cam):
    xyz_lidar = np.array([box[0], box[1], box[2], 1.0])
    xyz_cam = Tr_velo_to_cam @ xyz_lidar
    return [xyz_cam[0], xyz_cam[1], xyz_cam[2], *box[3:]]

def project_to_2d(box_cam, P2):
    return [0, 0, 50, 50]  # placeholder

# ==========================
# PROCESS DATA
# ==========================
frame_indices = [f.split('.')[0] for f in os.listdir(img_dir) if f.endswith('.jpg')]
frame_indices.sort()
annotations = []

for frame_id_str in frame_indices:
    img_path = os.path.join(img_dir, f"{frame_id_str}.jpg")
    radar_path = os.path.join(radar_dir, f"{frame_id_str}.bin")
    label_path = os.path.join(label_dir, f"{frame_id_str}.txt")
    calib_path = os.path.join(calib_dir, f"{frame_id_str}.txt")

    if not all([os.path.exists(p) for p in [img_path, radar_path, label_path, calib_path]]):
        print(f"Skipping frame {frame_id_str}: missing files")
        continue

    # Load/save image
    image = cv2.imread(img_path)
    out_img_path = os.path.join(image_out_dir, f"{frame_id_str}.jpg")
    cv2.imwrite(out_img_path, image)

    # Load radar and ensure 4 values per point
    radar_points = np.fromfile(radar_path, dtype=np.float32)
    if radar_points.size % 4 == 0:
        radar_points = radar_points.reshape(-1, 4)
    elif radar_points.size % 3 == 0:
        radar_points = radar_points.reshape(-1, 3)
        radar_points = np.hstack([radar_points, np.zeros((radar_points.shape[0], 1), dtype=np.float32)])
        print(f"Warning: radar file {os.path.basename(radar_path)} had 3 values per point, padded with 0 velocity")
    else:
        # If the number of floats is not divisible by 3 or 4, pad to nearest multiple of 4
        n_points = radar_points.size // 4
        remainder = radar_points.size % 4
        if remainder != 0:
            radar_points = np.pad(radar_points, (0, 4 - remainder), 'constant', constant_values=0)
        radar_points = radar_points.reshape(-1, 4)
        print(f"Warning: radar file {os.path.basename(radar_path)} had unexpected number of floats, padded to shape {radar_points.shape}")

    out_radar_path = os.path.join(radar_out_dir, f"{frame_id_str}.npy")
    np.save(out_radar_path, radar_points)

    # Load labels/calib
    objects = load_kitti_label(label_path)
    calib = load_kitti_calib(calib_path)
    P2 = calib['P2']
    Tr_velo_to_cam = np.vstack([calib['Tr_velo_to_cam'], [0, 0, 0, 1]])

    for obj in objects:
        obj['bbox_3d'] = lidar_to_camera(obj['bbox_3d'], Tr_velo_to_cam)
        obj['bbox_2d'] = project_to_2d(obj['bbox_3d'], P2)
        obj['radar'] = []

    annotations.append({
        "frame_id": int(frame_id_str),
        "image_path": out_img_path,
        "radar_path": out_radar_path,
        "objects": objects
    })

    if int(frame_id_str) % 100 == 0:
        print(f"Processed frame {frame_id_str}")

# Save annotations
with open(os.path.join(output_root, "annotations.json"), 'w') as f:
    json.dump(annotations, f, indent=2)

print("Dataset preprocessing complete!")