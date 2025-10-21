from dotenv import load_dotenv
load_dotenv()
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms, models
from Cnn import SimpleModel

# Import VoD classes (adjust your vod_repo path)
import sys

cwd = os.getcwd()
vod_repo_path = os.path.join(cwd, os.environ["VOD_REPO_PATH"])
sys.path.insert(0, vod_repo_path)

from vod.configuration import KittiLocations # type: ignore
from vod.frame import FrameDataLoader, FrameTransformMatrix, FrameLabels # type: ignore
from vod.visualization import Visualization2D # type: ignore
from vod.frame.transformations import homogeneous_coordinates, project_3d_to_2d, canvas_crop # type: ignore

# =========================
# Configuration
# =========================
images_root = os.path.join(cwd, os.environ["IMAGES_ROOT"])
annotations_path = os.path.join(cwd, os.environ["ANNOTATIONS_PATH"])
model_path = os.path.join(cwd, os.environ["MODELS_PATH"])

raw_root = os.path.join(cwd, os.environ["RAW_ROOT"])
output_root = os.path.join(cwd, os.environ["OUTPUT_ROOT"], "eval")

# Image normalization parameters
original_img_size = (1920, 1080)
max_3d_coord = 20.0

# =========================
# Dataset utilities
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleModel()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# =========================
# Load frame data
# =========================

frame_number = "00051"  # frame to visualize
kitti_locations = KittiLocations(root_dir=raw_root, output_dir=output_root)
frame_data = FrameDataLoader(kitti_locations=kitti_locations, frame_number=frame_number)
T = FrameTransformMatrix(frame_data)

# Load image
img_path = os.path.join(images_root, f"{frame_number}.png")
image = Image.open(img_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(device)

# =========================
# Run model prediction
# =========================
with torch.no_grad():
    pred_2d, pred_3d = model(image_tensor)
    pred_2d = pred_2d.cpu().numpy()[0]
    pred_3d = pred_3d.cpu().numpy()[0]

# Rescale predictions
pred_2d[0::2] *= original_img_size[0]  # x
pred_2d[1::2] *= original_img_size[1]  # y
pred_3d[:6] *= max_3d_coord  # x, y, z, h, w, l

# =========================
# Convert 3D bbox to corners
# =========================
def get_3d_corners(bbox3d):
    x, y, z, h, w, l, _ = bbox3d
    corners = np.array([
        [x - l/2, y - w/2, z - h/2],
        [x - l/2, y + w/2, z - h/2],
        [x + l/2, y + w/2, z - h/2],
        [x + l/2, y - w/2, z - h/2],
        [x - l/2, y - w/2, z + h/2],
        [x - l/2, y + w/2, z + h/2],
        [x + l/2, y + w/2, z + h/2],
        [x + l/2, y - w/2, z + h/2]
    ])
    return corners

pred_corners_3d = get_3d_corners(pred_3d)
corners_homo = homogeneous_coordinates(pred_corners_3d)
pred_corners_2d = project_3d_to_2d(corners_homo, T.camera_projection_matrix)
valid_idx = canvas_crop(pred_corners_2d, frame_data.image.shape, points_depth=pred_corners_3d[:, 2])
pred_corners_2d = pred_corners_2d[valid_idx]

# =========================
# Visualize
# =========================
fig, ax = plt.subplots(figsize=(10,6))
ax.imshow(frame_data.image)  # show original image

# Plot predicted 2D bounding box
if len(pred_2d) > 0:
    x_min, y_min = pred_2d[:2]
    x_max, y_max = pred_2d[2:]
    rect = plt.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, linewidth=2, edgecolor='r', facecolor='none', label="Pred 2D")
    ax.add_patch(rect)

# Plot projected 3D bbox
if len(pred_corners_2d) > 0:
    x_min, y_min = pred_corners_2d.min(axis=0)
    x_max, y_max = pred_corners_2d.max(axis=0)
    rect = plt.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, linewidth=2, edgecolor='b', facecolor='none', label="Pred 3D")
    ax.add_patch(rect)

# Optionally, show ground-truth using Visualization2D
vis2d = Visualization2D(frame_data)
fig_gt, ax_gt = vis2d.draw_plot(show_radar=False, show_gt=True, plot_figure=False, save_figure=False)
plt.show()
