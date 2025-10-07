import os
import sys
import numpy as np
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm

# ==========================
# ADD VOD REPO TO PYTHON PATH
# ==========================
vod_repo_path = r"C:\Users\Budge\OneDrive\Desktop\Schoolwork\Semester7\CSE486\CenterfusionVodTraining\centerfusion-vod-training\data\vod_repo\view_of_delft_dataset"
sys.path.insert(0, vod_repo_path)

from vod.configuration import KittiLocations #type: ignore
from vod.frame import FrameDataLoader, FrameTransformMatrix, FrameLabels # type: ignore
from vod.visualization import Visualization2D # type: ignore
from vod.frame.transformations import ( # type: ignore
    homogeneous_coordinates,
    homogeneous_transformation,
    project_3d_to_2d,
    canvas_crop
)

# ==========================
# CONFIG
# ==========================
raw_root = r"C:\Users\Budge\OneDrive\Desktop\Schoolwork\Semester7\CSE486\CenterfusionVodTraining\centerfusion-vod-training\data\View_of_Delft_dataset_PUBLIC\view_of_delft_PUBLIC\view_of_delft_PUBLIC"
output_root = r"C:\Users\Budge\OneDrive\Desktop\Schoolwork\Semester7\CSE486\CenterfusionVodTraining\centerfusion-vod-training\data\vod_processed"

image_out_dir = os.path.join(output_root, "images")
radar_out_dir = os.path.join(output_root, "radar")
radar2d_out_dir = os.path.join(output_root, "radar_2d")
os.makedirs(image_out_dir, exist_ok=True)
os.makedirs(radar_out_dir, exist_ok=True)
os.makedirs(radar2d_out_dir, exist_ok=True)

kitti_locations = KittiLocations(root_dir=raw_root, output_dir=output_root)

# ======================
# LOOP OVER FRAMES
# ======================
annotations = []
frame_numbers = [f"{i:05d}" for i in range(0, 9930)] # 9930 is the last frame in VoD dataset

for frame_number in tqdm(frame_numbers):
    try:
        frame_data = FrameDataLoader(kitti_locations=kitti_locations,
                                     frame_number=frame_number)

        # ========== Save overlayed image ==========
        vis2d = Visualization2D(frame_data)
        fig, ax = vis2d.draw_plot(show_radar=True, show_gt=True, plot_figure=False, save_figure=False)
        fig_save_path = os.path.join(image_out_dir, f"{frame_number}.png")
        fig.savefig(fig_save_path, bbox_inches='tight', transparent=True)
        plt.close(fig)

        # ========== Save radar calibrated data ==========
        radar_data = frame_data.radar_data

        T = FrameTransformMatrix(frame_data)
        radar_homo = homogeneous_coordinates(radar_data[:, :3])
        radar_cam = homogeneous_transformation(radar_homo, T.t_camera_radar)

        radar_out_path = os.path.join(radar_out_dir, f"{frame_number}.npy")
        np.save(radar_out_path, radar_cam[:, :3])

        # ========== Project radar to 2D ==========
        radar_2d = project_3d_to_2d(radar_cam, T.camera_projection_matrix)
        valid_idx = canvas_crop(radar_2d, frame_data.image.shape, points_depth=radar_cam[:, 2])
        radar_2d = radar_2d[valid_idx]
        radar2d_out_path = os.path.join(radar2d_out_dir, f"{frame_number}.npy")
        np.save(radar2d_out_path, radar_2d)

        # ========== Build annotations entry ==========
        frame_labels = FrameLabels(frame_data.raw_labels)
        frame_ann = {
            "frame_id": frame_number,
            "image_path": f"images/{frame_number}.png",
            "radar_path": f"radar/{frame_number}.npy",
            "radar_2d_path": f"radar_2d/{frame_number}.npy",
            "objects": []
        }

        # Compute 2D bounding boxes from 3D corners
        for obj in frame_labels.labels_dict:
            # Build 3D bounding box corners
            x, y, z = obj['x'], obj['y'], obj['z']
            h, w, l = obj['h'], obj['w'], obj['l']
            corners_3d = np.array([
                [x - l/2, y - w/2, z - h/2],
                [x - l/2, y + w/2, z - h/2],
                [x + l/2, y + w/2, z - h/2],
                [x + l/2, y - w/2, z - h/2],
                [x - l/2, y - w/2, z + h/2],
                [x - l/2, y + w/2, z + h/2],
                [x + l/2, y + w/2, z + h/2],
                [x + l/2, y - w/2, z + h/2]
            ])
            # Convert 3D corners to homogeneous coordinates
            corners_homo = homogeneous_coordinates(corners_3d)

            # Project corners directly to 2D using the camera projection matrix
            corners_2d = project_3d_to_2d(corners_homo, T.camera_projection_matrix)

            # Keep only points that lie inside the image
            valid_idx = canvas_crop(corners_2d, frame_data.image.shape, points_depth=corners_3d[:, 2])
            corners_2d = corners_2d[valid_idx]

            # Compute 2D bbox if points remain
            if len(corners_2d) > 0:
                x_min, y_min = corners_2d.min(axis=0)
                x_max, y_max = corners_2d.max(axis=0)
                bbox_2d = [int(x_min), int(y_min), int(x_max), int(y_max)]
            else:
                bbox_2d = None

            frame_ann["objects"].append({
                "class": obj['label_class'],
                "bbox_3d": {
                    "x": x, "y": y, "z": z,
                    "h": h, "w": w, "l": l,
                    "rotation": obj['rotation']
                },
                "bbox_2d": bbox_2d
            })

        annotations.append(frame_ann)

    except Exception as e:
        print(f"Skipping frame {frame_number} due to error: {e}")
        continue

# ======================
# SAVE ANNOTATIONS.JSON
# ======================
annotations_path = os.path.join(output_root, "annotations.json")
with open(annotations_path, "w") as f:
    json.dump(annotations, f, indent=2)

print(f"Preprocessing complete. Saved images, radar data, radar projections, and annotations.json to {output_root}")
