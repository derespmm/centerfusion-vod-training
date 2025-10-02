import os
import sys
import numpy as np
import json
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==========================
# ADD VOD REPO TO PYTHON PATH
# ==========================
vod_repo_path = os.path.join(os.path.dirname(__file__), "../data/vod_repo/view_of_delft_dataset")
sys.path.insert(0, vod_repo_path)

from vod.configuration import KittiLocations
from vod.frame import FrameDataLoader, FrameTransformMatrix
from vod.visualization import Visualization2D, Visualization3D

# ==========================
# CONFIG
# ==========================
# Raw VoD dataset root
raw_root = r"C:\Users\Budge\OneDrive\Desktop\Schoolwork\Semester7\CSE486\CenterfusionVodTraining\centerfusion-vod-training\data\View_of_Delft_dataset_PUBLIC\view_of_delft_PUBLIC\view_of_delft_PUBLIC"

# Output folder for processed data
output_root = os.path.join(os.path.dirname(__file__), "../data/vod_processed")

# Create output directories
image_out_dir = os.path.join(output_root, "images")
radar_out_dir = os.path.join(output_root, "radar")
radar2d_out_dir = os.path.join(output_root, "radar_2d")
os.makedirs(image_out_dir, exist_ok=True)
os.makedirs(radar_out_dir, exist_ok=True)
os.makedirs(radar2d_out_dir, exist_ok=True)

# Initialize KittiLocations (sets up all internal paths automatically)
kitti_locations = KittiLocations(root_dir=raw_root, output_dir=output_root)

# ======================
# LOOP OVER FRAMES
# ======================
annotations = []

frame_numbers = [f"{i:05d}" for i in range(0, 8682)]  # 8682 is the number of images in VoD training set

for frame_number in tqdm(frame_numbers):
    try:
        # Load frame
        frame_data = FrameDataLoader(kitti_locations=kitti_locations,
                                     frame_number=frame_number)

        # ========== Save overlayed image ==========
        img_out_path = os.path.join(image_out_dir, f"{frame_number}.jpg")
        vis2d = Visualization2D(frame_data)
        vis2d.draw_plot(show_radar=True, show_gt=True, plot_figure=False, save_figure=img_out_path)

        # ========== Save radar raw data ==========
        radar_data = frame_data.radar_data  # NxD numpy array
        radar_out_path = os.path.join(radar_out_dir, f"{frame_number}.npy")
        np.save(radar_out_path, radar_data)

        # ========== Project radar to 2D ==========
        T = FrameTransformMatrix(frame_data.calibration)
        radar_points_cam = T.radar_to_camera(radar_data)  # 3D cam coords
        radar_points_2d = T.camera_to_image(radar_points_cam)  # 2D pixels

        radar2d_out_path = os.path.join(radar2d_out_dir, f"{frame_number}.npy")
        np.save(radar2d_out_path, radar_points_2d)

        # ========== Build annotations entry ==========
        frame_ann = {
            "frame_id": frame_number,
            "image_path": f"images/{frame_number}.jpg",
            "radar_path": f"radar/{frame_number}.npy",
            "radar_2d_path": f"radar_2d/{frame_number}.npy",
            "objects": []
        }

        # Collect 3D labels
        for obj in frame_data.labels_3d:
            obj_dict = {
                "class": obj.obj_type,
                "bbox_3d": obj.as_dict(),  # dict with KITTI-like keys
                "bbox_2d": obj.bbox_2d if hasattr(obj, "bbox_2d") else None
            }
            frame_ann["objects"].append(obj_dict)

        annotations.append(frame_ann)

    except Exception as e:
        # Skip missing frames
        continue

# ======================
# SAVE ANNOTATIONS.JSON
# ======================
annotations_path = os.path.join(output_root, "annotations.json")
with open(annotations_path, "w") as f:
    json.dump(annotations, f, indent=2)

print(f"Preprocessing complete. Saved images, radar data, radar projections, and annotations.json to {output_root}")