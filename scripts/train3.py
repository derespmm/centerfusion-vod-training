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
MAX_OBJECTS = 20
ORIG_IMG_SIZE = (1936, 1216)
TRAIN_IMG_SIZE = (950, 604)


# ======================
# Dataset with Radar Fusion + Class IDs
# ======================
class RadarCameraDataset(Dataset):
    def __init__(self, annotations_path, images_root, radar2d_root, transform=None):
        with open(annotations_path, 'r') as f:
            self.annotations = json.load(f)
        self.images_root = images_root
        self.radar2d_root = radar2d_root
        self.transform = transform

        # Extract all unique classes
        class_set = set()
        for ann in self.annotations:
            for obj in ann["objects"]:
                class_set.add(obj["class"])
        self.classes_list = sorted(list(class_set))
        self.class_map = {cls: i for i, cls in enumerate(self.classes_list)}
        print(f"🔹 Found {len(self.classes_list)} classes: {self.classes_list}")

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

        img = Image.open(img_path).convert("RGB")
        img_resized = img.resize(IMG_SIZE)
        img_np = np.array(img_resized).astype(np.float32) / 255.0

        radar_map = self.make_radar_heatmap(radar_points, IMG_SIZE[::-1])
        radar_map = radar_map[..., np.newaxis]

        fused = np.concatenate([img_np, radar_map], axis=2)
        fused = torch.from_numpy(fused.transpose(2, 0, 1))

        # Initialize tensors
        num_objs = min(len(ann["objects"]), MAX_OBJECTS)
        bbox2d = torch.zeros((MAX_OBJECTS, 4), dtype=torch.float32)
        class_ids = torch.zeros(MAX_OBJECTS, dtype=torch.long)
        mask = torch.zeros(MAX_OBJECTS, dtype=torch.float32)

        scale_from_orig_to_train_x = TRAIN_IMG_SIZE[0] / ORIG_IMG_SIZE[0]
        scale_from_orig_to_train_y = TRAIN_IMG_SIZE[1] / ORIG_IMG_SIZE[1]
        scale_from_train_to_model_x = IMG_SIZE[0] / TRAIN_IMG_SIZE[0]
        scale_from_train_to_model_y = IMG_SIZE[1] / TRAIN_IMG_SIZE[1]

        for i, obj in enumerate(ann["objects"][:num_objs]):
            if obj["bbox_2d"]:
                b2d = np.array(obj["bbox_2d"], dtype=np.float32)
                b2d[0::2] *= scale_from_orig_to_train_x * scale_from_train_to_model_x
                b2d[1::2] *= scale_from_orig_to_train_y * scale_from_train_to_model_y
                b2d[0::2] /= IMG_SIZE[0]
                b2d[1::2] /= IMG_SIZE[1]
                bbox2d[i] = torch.from_numpy(np.clip(b2d, 0, 1))

            class_ids[i] = self.class_map.get(obj["class"], 0)
            mask[i] = 1.0

        return fused, bbox2d, class_ids, mask


# ======================
# Model with Classification + 2D Bbox
# ======================
class RadarCameraModel(nn.Module):
    def __init__(self, num_objects=MAX_OBJECTS, num_classes=14):
        super().__init__()
        base = models.resnet18(weights="IMAGENET1K_V1")
        base.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = base
        self.backbone.fc = nn.Identity()

        self.fc_2d = nn.Linear(512, num_objects * 4)
        self.fc_cls = nn.Linear(512, num_objects * num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        feat = self.backbone(x)
        bbox2d = self.fc_2d(feat).view(-1, MAX_OBJECTS, 4)
        cls_logits = self.fc_cls(feat).view(-1, MAX_OBJECTS, self.num_classes)
        return bbox2d, cls_logits


# ======================
# Loss Function
# ======================
def detection_loss(pred2d, gt2d, cls_logits, class_ids, mask):
    l1 = nn.SmoothL1Loss(reduction='none')
    loss2d = l1(pred2d, gt2d).sum(dim=-1)
    reg_loss = (loss2d * mask).mean()

    ce_loss_fn = nn.CrossEntropyLoss(reduction='none')
    cls_loss_per_obj = ce_loss_fn(
        cls_logits.view(-1, cls_logits.shape[-1]),
        class_ids.view(-1)
    )
    cls_loss_per_obj = cls_loss_per_obj.view(-1, MAX_OBJECTS)
    cls_loss = (cls_loss_per_obj * mask).mean()

    return reg_loss + cls_loss, reg_loss.item(), cls_loss.item()


# ======================
# Training Loop
# ======================
def train_model():
    cwd = os.getcwd()
    data_root = os.path.join(cwd, "data", "vod_processed")
    dataset = RadarCameraDataset(
        annotations_path=os.path.join(data_root, "annotations.json"),
        images_root=os.path.join(data_root, "images"),
        radar2d_root=os.path.join(data_root, "radar_2d")
    )

    # User prompts
    epochs_input = input(f"Enter number of epochs (default 5): ")
    num_epochs = int(epochs_input) if epochs_input.strip() else 5

    batch_input = input(f"Enter batch size (default 24): ")
    batch_size = int(batch_input) if batch_input.strip() else 24

    load_existing = input("Load existing model? (y/n, default n): ").strip().lower() == 'y'

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RadarCameraModel(num_classes=len(dataset.classes_list)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    if load_existing and os.path.exists("models/radar_camera_fusion.pth"):
        model.load_state_dict(torch.load("models/radar_camera_fusion.pth", map_location=device))
        print("✅ Loaded existing model.")

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_reg_loss, total_cls_loss = 0, 0, 0

        for i, (fused, b2d, class_ids, mask) in enumerate(train_loader):
            fused, b2d, class_ids, mask = (
                fused.to(device),
                b2d.to(device),
                class_ids.to(device),
                mask.to(device)
            )

            optimizer.zero_grad()
            p2d, cls_logits = model(fused)
            loss, reg_loss_val, cls_loss_val = detection_loss(p2d, b2d, cls_logits, class_ids, mask)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_reg_loss += reg_loss_val
            total_cls_loss += cls_loss_val

            if i % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {i}, "
                      f"Total Loss={total_loss/(i+1):.4f}, "
                      f"Reg Loss={total_reg_loss/(i+1):.4f}, "
                      f"Cls Loss={total_cls_loss/(i+1):.4f}")

        print(f"✅ Epoch {epoch+1} completed. Avg Loss: {total_loss/len(train_loader):.4f}")

        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/radar_camera_fusion.pth")
        print("Model saved.")


if __name__ == "__main__":
    train_model()
