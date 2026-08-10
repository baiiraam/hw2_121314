"""
Bonus: Inference-only run of pretrained torchvision models.
Run on 3-5 of your own photos placed in data/samples/.
"""
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patches
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    keypointrcnn_resnet50_fpn,
    maskrcnn_resnet50_fpn,
)
from loguru import logger


def load_image(image_path, device):
    """Load and preprocess image for inference."""
    logger.debug(f"Loading image: {image_path}")
    img = Image.open(image_path).convert("RGB")
    img_original = np.array(img)
    transform = transforms.ToTensor()
    img_tensor = transform(img).to(device)
    logger.debug(f"Image loaded: {img_original.shape}, tensor: {img_tensor.shape}")
    return img_tensor, img_original


def run_detection(image_path, device, score_threshold=0.5):
    """Bonus (a): Faster R-CNN with FPN backbone."""
    logger.info(f"Running detection on: {os.path.basename(image_path)}")

    start_time = time.time()

    model = fasterrcnn_resnet50_fpn(weights="DEFAULT").to(device)
    model.eval()

    img_tensor, img_original = load_image(image_path, device)

    with torch.no_grad():
        predictions = model([img_tensor])

    pred = predictions[0]
    boxes = pred["boxes"].cpu().numpy()
    labels = pred["labels"].cpu().numpy()
    scores = pred["scores"].cpu().numpy()

    coco_names = {
        1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
        6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic_light",
        11: "fire_hydrant", 13: "stop_sign", 14: "parking_meter", 15: "bench",
        16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep",
        21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe",
        27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie",
        33: "suitcase", 34: "frisbee", 35: "skis", 36: "snowboard",
        37: "sports_ball", 38: "kite", 39: "baseball_bat", 40: "baseball_glove",
        41: "skateboard", 42: "surfboard", 43: "tennis_racket",
        44: "bottle", 46: "wine_glass", 47: "cup", 48: "fork",
        49: "knife", 50: "spoon", 51: "bowl", 52: "banana",
        53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli",
        57: "carrot", 58: "hot_dog", 59: "pizza", 60: "donut",
        61: "cake", 62: "chair", 63: "couch", 64: "potted_plant",
        65: "bed", 67: "dining_table", 70: "toilet", 72: "tv",
        73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard",
        77: "cell_phone", 78: "microwave", 79: "oven", 80: "toaster",
        81: "sink", 82: "refrigerator", 84: "book", 85: "clock",
        86: "vase", 87: "scissors", 88: "teddy_bear", 89: "hair_drier",
        90: "toothbrush",
    }

    valid_indices = scores >= score_threshold
    boxes = boxes[valid_indices]
    labels = labels[valid_indices]
    scores = scores[valid_indices]

    logger.info(f"Found {len(boxes)} detections above threshold {score_threshold}")

    _, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img_original)
    ax.set_title(f"Detection: {os.path.basename(image_path)}")

    for box, label, score in zip(boxes, labels, scores):
        if score < score_threshold:
            continue
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="r", facecolor="none"
        )
        ax.add_patch(rect)
        label_name = coco_names.get(label, f"class_{label}")
        ax.text(
            x1, y1 - 5, f"{label_name}: {score:.2f}",
            color="white", fontsize=8,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "red", "alpha": 0.7},
        )

    ax.text(
        0.5, -0.05,
        "NMS applied by model post-processing",
        transform=ax.transAxes, ha="center", fontsize=10, style="italic",
    )

    plt.tight_layout()
    output_path = f"figures/zoo_detection_{os.path.basename(image_path).split('.')[0]}.png"
    os.makedirs("figures", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    elapsed = time.time() - start_time
    logger.success(f"Detection result saved to {output_path} in {elapsed:.2f}s")


def run_segmentation(image_path, device, score_threshold=0.5):
    """Bonus (b): Mask R-CNN with instance masks."""
    logger.info(f"Running segmentation on: {os.path.basename(image_path)}")

    start_time = time.time()

    model = maskrcnn_resnet50_fpn(weights="DEFAULT").to(device)
    model.eval()

    img_tensor, img_original = load_image(image_path, device)

    with torch.no_grad():
        predictions = model([img_tensor])

    pred = predictions[0]
    boxes = pred["boxes"].cpu().numpy()
    labels = pred["labels"].cpu().numpy()
    scores = pred["scores"].cpu().numpy()
    masks = pred["masks"].cpu().numpy()

    valid_indices = scores >= score_threshold
    boxes = boxes[valid_indices]
    labels = labels[valid_indices]
    scores = scores[valid_indices]
    masks = masks[valid_indices]

    logger.info(f"Found {len(boxes)} segmentations above threshold {score_threshold}")

    _, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img_original)
    ax.set_title(f"Segmentation: {os.path.basename(image_path)}")

    colors = plt.cm.tab20(np.linspace(0, 1, len(masks)))

    for i, (box, label, score, mask) in enumerate(zip(boxes, labels, scores, masks)):
        if score < score_threshold:
            continue

        mask_binary = mask[0] > 0.5
        mask_overlay = np.zeros_like(img_original, dtype=np.uint8)
        mask_overlay[mask_binary] = (colors[i][:3] * 255).astype(np.uint8)

        ax.imshow(mask_overlay, alpha=0.3)

        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=colors[i], facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            x1, y1 - 5, f"Instance {i}: {score:.2f}",
            color="white", fontsize=8,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": colors[i], "alpha": 0.7},
        )

    plt.tight_layout()
    output_path = f"figures/zoo_segmentation_{os.path.basename(image_path).split('.')[0]}.png"
    os.makedirs("figures", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    elapsed = time.time() - start_time
    logger.success(f"Segmentation result saved to {output_path} in {elapsed:.2f}s")


def run_pose(image_path, device, score_threshold=0.5):
    """Bonus (c): Keypoint R-CNN for human pose estimation."""
    logger.info(f"Running pose estimation on: {os.path.basename(image_path)}")

    start_time = time.time()

    model = keypointrcnn_resnet50_fpn(weights="DEFAULT").to(device)
    model.eval()

    img_tensor, img_original = load_image(image_path, device)

    with torch.no_grad():
        predictions = model([img_tensor])

    pred = predictions[0]

    if "keypoints" not in pred or len(pred["keypoints"]) == 0:
        logger.warning("No keypoints detected in this image.")
        return

    keypoints = pred["keypoints"].cpu().numpy()
    scores = pred["scores"].cpu().numpy()
    boxes = pred["boxes"].cpu().numpy()

    valid_indices = scores >= score_threshold

    if not any(valid_indices):
        logger.warning("No valid keypoints above threshold.")
        return

    keypoints = keypoints[valid_indices]
    scores = scores[valid_indices]
    boxes = boxes[valid_indices]

    logger.info(f"Found {len(keypoints)} person(s) with keypoints above threshold {score_threshold}")

    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7),
        (6, 8), (7, 9), (8, 10), (5, 11), (6, 12), (11, 13),
        (12, 14), (13, 15), (14, 16),
    ]

    joint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    ]

    _, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img_original)
    ax.set_title(f"Pose Estimation: {os.path.basename(image_path)}")

    for person_idx, (kp, score) in enumerate(zip(keypoints, scores)):
        if score < score_threshold:
            continue

        visible = kp[:, 2] > 0.5
        kp_xy = kp[:, :2]

        for start, end in skeleton:
            if visible[start] and visible[end]:
                ax.plot(
                    [kp_xy[start, 0], kp_xy[end, 0]],
                    [kp_xy[start, 1], kp_xy[end, 1]],
                    "b-", linewidth=2, alpha=0.7,
                )

        for joint_idx, (x, y, vis) in enumerate(kp):
            if vis > 0.5:
                ax.scatter(x, y, c="r", s=30, zorder=5)
                if joint_idx in [5, 6, 7, 8, 9, 10]:
                    ax.text(
                        x + 5, y - 5, joint_names[joint_idx][:3],
                        fontsize=6, color="yellow",
                        bbox={"boxstyle": "round,pad=0.2", "facecolor": "black", "alpha": 0.5},
                    )

        if len(boxes) > person_idx:
            x1, y1, x2, y2 = boxes[person_idx]
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor="g", facecolor="none",
            )
            ax.add_patch(rect)

    ax.text(
        0.5, -0.05,
        f"Keypoints detected: {len(keypoints)} person(s) (score threshold: {score_threshold})",
        transform=ax.transAxes, ha="center", fontsize=10, style="italic",
    )

    plt.tight_layout()
    output_path = f"figures/zoo_pose_{os.path.basename(image_path).split('.')[0]}.png"
    os.makedirs("figures", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    elapsed = time.time() - start_time
    logger.success(f"Pose result saved to {output_path} in {elapsed:.2f}s")


def main():
    """Run all three models on sample images."""
    logger.info("=" * 50)
    logger.info("Bonus: Inference on Sample Images")
    logger.info("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    os.makedirs("data/samples", exist_ok=True)

    sample_images = [
        f for f in os.listdir("data/samples")
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not sample_images:
        logger.warning("No sample images found in data/samples/.")
        logger.info("Please add 3-5 of your own photos to data/samples/")
        return

    sample_images = sample_images[:5]
    logger.info(f"Found {len(sample_images)} sample images")

    for img_file in sample_images:
        img_path = os.path.join("data/samples", img_file)
        run_detection(img_path, device)
        run_segmentation(img_path, device)
        run_pose(img_path, device)

    logger.success("\nBonus inference complete!")
    logger.success("Outputs saved to figures/zoo_*.png")


if __name__ == "__main__":
    main()