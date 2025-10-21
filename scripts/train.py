import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np

# ======================
# Dataset
# ======================
class BoundingBoxDataset(Dataset):
    def __init__(self, annotations_path, images_root, transform=None, img_size=(224, 224), max_3d_coord=20.0):
        """
        annotations_path: path to annotations.json
        images_root: path to folder containing images
        transform: torchvision transforms for images
        img_size: (width, height) used for normalization of 2D boxes
        max_3d_coord: scaling factor for 3D boxes
        """
        with open(annotations_path, 'r') as f:
            self.annotations = json.load(f)

        self.images_root = images_root
        self.transform = transform
        self.img_w, self.img_h = img_size
        self.max_3d = max_3d_coord

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]

        # Load image
        image_path = os.path.join(self.images_root, os.path.basename(ann["image_path"]))
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Take first object only (extendable for multi-object training)
        obj = ann["objects"][0] if ann["objects"] else None
        if obj is None:
            bbox_2d = np.zeros(4, dtype=np.float32)
            bbox_3d = np.zeros(7, dtype=np.float32)
        else:
            # --- Normalize 2D bbox ---
            if obj["bbox_2d"]:
                bbox_2d = np.array(obj["bbox_2d"], dtype=np.float32)
                # Normalize x by image width, y by image height
                bbox_2d[0::2] /= self.img_w  # x_min, x_max
                bbox_2d[1::2] /= self.img_h  # y_min, y_max
                # Clip to [0,1] just in case
                bbox_2d = np.clip(bbox_2d, 0.0, 1.0)
            else:
                bbox_2d = np.zeros(4, dtype=np.float32)

            # --- Normalize 3D bbox ---
            b3d = obj["bbox_3d"]
            bbox_3d = np.array([b3d["x"], b3d["y"], b3d["z"],
                                b3d["h"], b3d["w"], b3d["l"],
                                b3d["rotation"]], dtype=np.float32)
            bbox_3d[:6] /= self.max_3d  # scale x,y,z,h,w,l
            # Optionally clip very large/small values
            bbox_3d[:6] = np.clip(bbox_3d[:6], 0.0, 1.0)
            # rotation stays in radians (assume -pi to pi)
            bbox_3d[6] = np.clip(bbox_3d[6], -np.pi, np.pi)

        return image, torch.from_numpy(bbox_2d), torch.from_numpy(bbox_3d)


# ======================
# Transform & Dataloader
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

cwd = os.getcwd()
annotations_path = os.path.join(cwd, r"data\vod_processed\annotations.json")
images_root = os.path.join(cwd, r"data\vod_processed\images")
dataset = BoundingBoxDataset(
    annotations_path=annotations_path,
    images_root=images_root,
    transform=transform
)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# ======================
# Simple CNN model
# ======================
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Use a pretrained backbone
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Identity()  # Remove final classification
        self.fc_2d = nn.Linear(512, 4)
        self.fc_3d = nn.Linear(512, 7)

    def forward(self, x):
        features = self.backbone(x)
        bbox_2d = self.fc_2d(features)
        bbox_3d = self.fc_3d(features)
        return bbox_2d, bbox_3d

model = SimpleModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ======================
# Training loop
# ======================
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(5):  # small number for example
    model.train()
    total_loss = 0
    for batch_idx, (imgs, bbox2d, bbox3d) in enumerate(dataloader):
        
        if batch_idx >= 100:  # <-- stop after 100 batches
            break

        imgs = imgs.to(device)
        bbox2d = bbox2d.to(device)
        bbox3d = bbox3d.to(device)

        optimizer.zero_grad()
        pred_2d, pred_3d = model(imgs)
        loss = criterion(pred_2d, bbox2d) + criterion(pred_3d, bbox3d)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # ===== Debugging / live output =====
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1} finished, Average Loss: {total_loss/len(dataloader):.4f}")


print("Training complete!")

# ======================
# Save the trained model
# ======================
save_dir = os.path.join(cwd, "models")
os.makedirs(save_dir, exist_ok=True)

model_save_path = os.path.join(save_dir, "simple_model.pth")
torch.save(model.state_dict(), model_save_path)

print(f"✅ Model saved to: {model_save_path}")

