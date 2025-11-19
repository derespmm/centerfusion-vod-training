import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import random

# Constants (from your config)
IMG_SIZE = (224, 224)
MAX_OBJECTS = 20

@torch.no_grad()
def evaluate_map50(model, loader, device, iou_thresh=0.5, score_thresh=0.3):
    """
    Evaluate mAP@0.5 for your RadarCameraModel.
    """

    model.eval()
    all_tp, all_fp, all_scores, num_gt = [], [], [], 0

    def box_iou_numpy(boxes1, boxes2):
        """
        Compute IoU between two sets of boxes (x1,y1,x2,y2)
        """
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        inter_x1 = np.maximum(boxes1[:, None, 0], boxes2[:, 0])
        inter_y1 = np.maximum(boxes1[:, None, 1], boxes2[:, 1])
        inter_x2 = np.minimum(boxes1[:, None, 2], boxes2[:, 2])
        inter_y2 = np.minimum(boxes1[:, None, 3], boxes2[:, 3])
        inter_w = np.clip(inter_x2 - inter_x1, 0, None)
        inter_h = np.clip(inter_y2 - inter_y1, 0, None)
        inter_area = inter_w * inter_h
        union = area1[:, None] + area2 - inter_area
        return inter_area / np.clip(union, 1e-6, None)

    for fused, gt_bboxes, gt_labels, mask in tqdm(loader, desc="Eval @0.5 IoU"):
        fused, gt_bboxes, gt_labels, mask = (
            fused.to(device),
            gt_bboxes.to(device),
            gt_labels.to(device),
            mask.to(device)
        )

        pred_bboxes, cls_logits = model(fused)
        probs = torch.softmax(cls_logits, dim=-1)
        scores, labels = probs.max(dim=-1)

        for i in range(fused.size(0)):  # iterate per image
            gt_mask = mask[i] > 0
            gt_boxes = gt_bboxes[i][gt_mask].cpu().numpy()
            gt_labels_i = gt_labels[i][gt_mask].cpu().numpy()
            num_gt += len(gt_boxes)

            pred_boxes = pred_bboxes[i].cpu().numpy()
            pred_labels = labels[i].cpu().numpy()
            pred_scores = scores[i].cpu().numpy()

            keep = pred_scores >= score_thresh
            pred_boxes, pred_labels, pred_scores = (
                pred_boxes[keep], pred_labels[keep], pred_scores[keep]
            )

            # Denormalize [0,1] to pixel coords
            pred_boxes = pred_boxes * np.array([IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[0], IMG_SIZE[1]])
            gt_boxes = gt_boxes * np.array([IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[0], IMG_SIZE[1]])

            used = np.zeros(len(gt_boxes), dtype=bool)
            order = np.argsort(-pred_scores)

            for j in order:
                box_p = pred_boxes[j][None, :]
                label_p = pred_labels[j]
                if len(gt_boxes) == 0:
                    all_tp.append(0); all_fp.append(1)
                    all_scores.append(float(pred_scores[j]))
                    continue

                ious = box_iou_numpy(box_p, gt_boxes)[0]
                ious[gt_labels_i != label_p] = 0  # only same class

                best_iou_idx = np.argmax(ious)
                best_iou = ious[best_iou_idx]

                if best_iou >= iou_thresh and not used[best_iou_idx]:
                    all_tp.append(1)
                    all_fp.append(0)
                    used[best_iou_idx] = True
                else:
                    all_tp.append(0)
                    all_fp.append(1)
                all_scores.append(float(pred_scores[j]))

    if not all_scores:
        return {"AP@0.5": 0.0, "Precision": 0.0, "Recall": 0.0}

    order = np.argsort(-np.array(all_scores))
    tp = np.array(all_tp)[order]
    fp = np.array(all_fp)[order]
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / max(1, num_gt)
    precision = tp_cum / np.maximum(1, tp_cum + fp_cum)

    ap = 0.0
    for r in np.linspace(0, 1, 11):
        p = precision[recall >= r].max() if np.any(recall >= r) else 0
        ap += p
    ap /= 11.0

    return {
        "AP@0.5": float(ap),
        "Precision": float(tp_cum[-1] / max(1, (tp_cum[-1] + fp_cum[-1]))),
        "Recall": float(tp_cum[-1] / max(1, num_gt)),
    }


@torch.no_grad()
def show_prediction(model, dataset, device, idx=None, score_thresh=0.3):
    """
    Visualize predicted and GT boxes for a single sample.
    """

    model.eval()
    if idx is None:
        idx = random.randint(0, len(dataset) - 1)

    fused, gt_bboxes, gt_labels, mask = dataset[idx]
    img_np = fused[:3].permute(1, 2, 0).numpy()  # ignore radar channel for display

    pred_bboxes, cls_logits = model(fused.unsqueeze(0).to(device))
    probs = torch.softmax(cls_logits, dim=-1)
    scores, labels = probs[0].max(dim=-1)
    boxes = pred_bboxes[0].cpu().numpy()
    scores = scores.cpu().numpy()
    labels = labels.cpu().numpy()

    keep = scores >= score_thresh
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    boxes = boxes * np.array([IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[0], IMG_SIZE[1]])

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(img_np)
    ax.axis('off')

    # Draw predictions
    for (x1, y1, x2, y2), sc, lab in zip(boxes, scores, labels):
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                 linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        cls_name = dataset.classes_list[lab]
        ax.text(x1, max(0, y1 - 5), f"{cls_name} {sc:.2f}",
                color='black', fontsize=9,
                bbox=dict(facecolor='lime', alpha=0.4, edgecolor='none'))

    # Draw GT boxes
    gt_mask = mask > 0
    gt_boxes = gt_bboxes[gt_mask].numpy() * np.array([IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[0], IMG_SIZE[1]])
    gt_labs = gt_labels[gt_mask].numpy()
    for (x1, y1, x2, y2), lab in zip(gt_boxes, gt_labs):
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                 linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
        ax.add_patch(rect)
        cls_name = dataset.classes_list[lab]
        ax.text(x1, max(0, y1 - 15), f"GT: {cls_name}",
                color='white', fontsize=9,
                bbox=dict(facecolor='red', alpha=0.4, edgecolor='none'))

    plt.show()

def __main__():
    import os
    from torch.utils.data import DataLoader
    from train3 import RadarCameraDataset, RadarCameraModel

    # Load your dataset
    data_root = r"C:\Users\Budge\OneDrive\Desktop\Schoolwork\Semester7\CSE486\CenterfusionVodTraining\centerfusion-vod-training\data\vod_processed"
    dataset = RadarCameraDataset(
        annotations_path=os.path.join(data_root, "annotations.json"),
        images_root=os.path.join(data_root, "images"),
        radar2d_root=os.path.join(data_root, "radar_2d")
    )

    # Prepare model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RadarCameraModel(num_classes=len(dataset.classes_list)).to(device)
    model.load_state_dict(torch.load("models/radar_camera_fusion.pth", map_location=device))

    # Evaluation
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    metrics = evaluate_map50(model, loader, device)
    print(metrics)

    # Visualization
    show_prediction(model, dataset, device, idx=42, score_thresh=0.3)

if __name__ == "__main__":
    __main__()