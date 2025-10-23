import os, json, torch, numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from PIL import Image
import cv2

# ======================
# CONFIG
# ======================
IMG_SIZE = (224, 224)
MAX_OBJECTS = 10
MAX_3D = 20.0
ORIG_IMG_SIZE = (1936, 1216)  # original camera frame
TRAIN_IMG_SIZE = (950, 604)   # your overlayed images (used for training)

# ======================
# Dataset with Radar Fusion
# ======================
class RadarCameraDataset(Dataset):
    def __init__(self, annotations_path, images_root, radar2d_root, transform=None):
        with open(annotations_path, 'r') as f:
            self.annotations = json.load(f)
        self.images_root = images_root
        self.radar2d_root = radar2d_root
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def make_radar_heatmap(self, radar_points, img_size):
        h, w = img_size
        heatmap = np.zeros((h, w), dtype=np.float32)
        for (u, v) in radar_points.astype(int):
            if 0 <= v < h and 0 <= u < w:
                cv2.circle(heatmap, (u, v), 2, 1.0, -1)
        heatmap = cv2.GaussianBlur(heatmap, (7, 7), 1)
        return heatmap / heatmap.max() if heatmap.max() > 0 else heatmap

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_path = os.path.join(self.images_root, os.path.basename(ann["image_path"]))
        radar_path = os.path.join(self.radar2d_root, os.path.basename(ann["radar_2d_path"]))
        radar_points = np.load(radar_path) if os.path.exists(radar_path) else np.empty((0, 2))

        # Load training image (950x604)
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size  # should be 950x604

        # Resize to model input (224x224)
        img_resized = img.resize(IMG_SIZE)
        img_np = np.array(img_resized).astype(np.float32) / 255.0

        # Create radar heatmap (scaled to IMG_SIZE)
        radar_map = self.make_radar_heatmap(radar_points, IMG_SIZE[::-1])
        radar_map = radar_map[..., np.newaxis]

        # Fuse RGB + radar
        fused = np.concatenate([img_np, radar_map], axis=2)
        fused = torch.from_numpy(fused.transpose(2, 0, 1))

        # Initialize tensors
        num_objs = min(len(ann["objects"]), MAX_OBJECTS)
        bbox2d = torch.zeros((MAX_OBJECTS, 4), dtype=torch.float32)
        bbox3d = torch.zeros((MAX_OBJECTS, 7), dtype=torch.float32)
        mask = torch.zeros(MAX_OBJECTS, dtype=torch.float32)

        # Scale factors
        scale_from_orig_to_train_x = TRAIN_IMG_SIZE[0] / ORIG_IMG_SIZE[0]
        scale_from_orig_to_train_y = TRAIN_IMG_SIZE[1] / ORIG_IMG_SIZE[1]
        scale_from_train_to_model_x = IMG_SIZE[0] / TRAIN_IMG_SIZE[0]
        scale_from_train_to_model_y = IMG_SIZE[1] / TRAIN_IMG_SIZE[1]

        for i, obj in enumerate(ann["objects"][:num_objs]):
            if obj["bbox_2d"]:
                b2d = np.array(obj["bbox_2d"], dtype=np.float32)

                # Step 1: convert from original (1936x1216) to training (950x604)
                b2d[0::2] *= scale_from_orig_to_train_x
                b2d[1::2] *= scale_from_orig_to_train_y

                # Step 2: convert from training (950x604) to model input (224x224)
                b2d[0::2] *= scale_from_train_to_model_x
                b2d[1::2] *= scale_from_train_to_model_y

                # Step 3: normalize [0,1]
                b2d[0::2] /= IMG_SIZE[0]
                b2d[1::2] /= IMG_SIZE[1]

                bbox2d[i] = torch.from_numpy(np.clip(b2d, 0, 1))

            # Normalize 3D boxes
            b3d = obj["bbox_3d"]
            b3d = np.array([
                b3d["x"], b3d["y"], b3d["z"],
                b3d["h"], b3d["w"], b3d["l"],
                b3d["rotation"]
            ], dtype=np.float32)
            b3d[:6] /= MAX_3D
            b3d[:6] = np.clip(b3d[:6], 0.0, 1.0)
            bbox3d[i] = torch.from_numpy(b3d)
            mask[i] = 1.0

        return fused, bbox2d, bbox3d, mask


# ======================
# Model
# ======================
class RadarCameraModel(nn.Module):
    def __init__(self, num_objects=MAX_OBJECTS):
        super().__init__()
        base = models.resnet18(weights="IMAGENET1K_V1")
        base.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = base
        self.backbone.fc = nn.Identity()

        self.fc_2d = nn.Linear(512, num_objects * 4)
        self.fc_3d = nn.Linear(512, num_objects * 7)

    def forward(self, x):
        feat = self.backbone(x)
        bbox2d = self.fc_2d(feat).view(-1, MAX_OBJECTS, 4)
        bbox3d = self.fc_3d(feat).view(-1, MAX_OBJECTS, 7)
        return bbox2d, bbox3d


# ======================
# Loss Function
# ======================
def detection_loss(pred2d, gt2d, pred3d, gt3d, mask):
    l1 = nn.SmoothL1Loss(reduction='none')
    loss2d = l1(pred2d, gt2d).sum(dim=-1)
    loss3d = l1(pred3d, gt3d).sum(dim=-1)
    loss = (loss2d + loss3d) * mask
    return loss.mean()


# ======================
# Training
# ======================
def train_model():
    cwd = os.getcwd()
    data_root = os.path.join(cwd, "data", "vod_processed")
    dataset = RadarCameraDataset(
        annotations_path=os.path.join(data_root, "annotations.json"),
        images_root=os.path.join(data_root, "images"),
        radar2d_root=os.path.join(data_root, "radar_2d")
    )
    dataloader = DataLoader(dataset, batch_size=24, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RadarCameraModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(5):
        model.train()
        total_loss = 0
        for i, (fused, b2d, b3d, mask) in enumerate(dataloader):
            fused, b2d, b3d, mask = fused.to(device), b2d.to(device), b3d.to(device), mask.to(device)
            optimizer.zero_grad()
            p2d, p3d = model(fused)
            loss = detection_loss(p2d, b2d, p3d, b3d, mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if i % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1} avg loss: {total_loss/len(dataloader):.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/radar_camera_fusion.pth")
    print("✅ Model saved!")


if __name__ == "__main__":
    train_model()
