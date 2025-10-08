import os
import sys
import numpy as np
import json
import shutil
import matplotlib.pyplot as plt

# ==========================
# ADD VOD REPO TO PYTHON PATH
# ==========================
vod_repo_path = r"C:\Users\Budge\OneDrive\Desktop\Schoolwork\Semester7\CSE486\CenterfusionVodTraining\centerfusion-vod-training\data\vod_repo\view_of_delft_dataset"
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
output_root = r"C:\Users\Budge\OneDrive\Desktop\Schoolwork\Semester7\CSE486\CenterfusionVodTraining\centerfusion-vod-training\data\vod_processed"
image_out_dir = os.path.join(output_root, "images")
radar_out_dir = os.path.join(output_root, "radar")
os.makedirs(image_out_dir, exist_ok=True)
os.makedirs(radar_out_dir, exist_ok=True)

# Initialize KittiLocations (sets up all internal paths automatically)
kitti_locations = KittiLocations(root_dir=raw_root, output_dir=output_root)

# Example: Load a specific frame
frame_data = FrameDataLoader(kitti_locations=kitti_locations, frame_number="00005")

print(frame_data.radar_data)

# 3D Visualization
vis_2d = Visualization2D(frame_data)
vis_2d.draw_plot(show_gt=True, show_radar=True)