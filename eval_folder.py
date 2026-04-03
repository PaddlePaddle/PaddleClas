import sys

sys.path.append(
    "/media/user/40da1a25-a924-43a2-b274-e33d39ea8680/cv/lzy/paddle_hiera-main"
)
import os
import warnings

import paddle
from paddle_utils import *

warnings.filterwarnings("ignore")
import hiera
from PIL import Image
from tqdm import tqdm

MODEL_CONFIGS = {
    "hiera_tiny_224": {"target_acc": 82.8},
    "hiera_small_224": {"target_acc": 83.8},
    "hiera_base_224": {"target_acc": 84.5},
    "hiera_base_plus_224": {"target_acc": 85.2},
    "hiera_large_224": {"target_acc": 86.1},
    "hiera_huge_224": {"target_acc": 86.9},
}


def get_model(model_name):
    print(f"Loading {model_name}...")
    if model_name == "hiera_tiny_224":
        model = hiera.hiera_tiny_224(pretrained=True, checkpoint="mae_in1k_ft_in1k")
    elif model_name == "hiera_small_224":
        model = hiera.hiera_small_224(pretrained=True, checkpoint="mae_in1k_ft_in1k")
    elif model_name == "hiera_base_224":
        model = hiera.hiera_base_224(pretrained=True, checkpoint="mae_in1k_ft_in1k")
    elif model_name == "hiera_base_plus_224":
        model = hiera.hiera_base_plus_224(
            pretrained=True, checkpoint="mae_in1k_ft_in1k"
        )
    elif model_name == "hiera_large_224":
        model = hiera.hiera_large_224(pretrained=True, checkpoint="mae_in1k_ft_in1k")
    elif model_name == "hiera_huge_224":
        model = hiera.hiera_huge_224(pretrained=True, checkpoint="mae_in1k_ft_in1k")
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model


def evaluate_imagenet_folder(model_name, data_dir, batch_size=32):
    device = paddle.device("cuda" if paddle.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = get_model(model_name)
    model = model.to(device)
    model.eval()
    target_acc = MODEL_CONFIGS[model_name]["target_acc"]
    transform = paddle.vision.transforms.Compose(
        [
            paddle.vision.transforms.Resize(size=256),
            paddle.vision.transforms.CenterCrop(224),
            paddle.vision.transforms.ToTensor(),
            paddle.vision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    print(f"\nLoading dataset from {data_dir}...")
    class_dirs = []
    for d in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, d)):
            try:
                class_dirs.append(int(d))
            except:
                pass
    class_dirs = sorted(class_dirs)
    print(f"Found {len(class_dirs)} classes")
    all_images = []
    all_labels = []
    for class_idx, class_dir in enumerate(class_dirs):
        class_path = os.path.join(data_dir, str(class_dir))
        images = [
            f
            for f in os.listdir(class_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        for img_name in images:
            all_images.append(os.path.join(class_path, img_name))
            all_labels.append(class_idx)
    print(f"Total images: {len(all_images)}")
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    with paddle.no_grad():
        for i in tqdm(range(0, len(all_images), batch_size), desc="Evaluating"):
            batch_images = all_images[i : i + batch_size]
            batch_labels = all_labels[i : i + batch_size]
            images = []
            labels = []
            for img_path, label in zip(batch_images, batch_labels):
                try:
                    img = Image.open(img_path).convert("RGB")
                    img = transform(img)
                    images.append(img)
                    labels.append(label)
                except Exception as e:
                    continue
            if not images:
                continue
            images = paddle.stack(images).to(device)
            labels = paddle.tensor(labels).to(device)
            outputs = model(images)
            _, top1 = outputs._max(1)
            _, top5 = outputs.topk(5, 1, True, True)
            total += labels.size(0)
            correct_top1 += top1.eq(labels).sum().item()
            correct_top5 += top5.eq(labels.view(-1, 1).expand_as(top5)).sum().item()
    if total == 0:
        print("ERROR: No images were processed!")
        return
    top1_acc = 100.0 * correct_top1 / total
    top5_acc = 100.0 * correct_top5 / total
    print("\n" + "=" * 60)
    print(f"Model: {model_name}")
    print(f"Target Top-1: {target_acc}%")
    print(f"Actual Top-1: {top1_acc:.2f}%")
    print(f"Top-5: {top5_acc:.2f}%")
    print(f"Total images: {total}")
    print(f"Gap: {top1_acc - target_acc:+.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Hiera on ImageNet")
    parser.add_argument("--model", type=str, default="hiera_tiny_224")
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    evaluate_imagenet_folder(args.model, args.data_dir, args.batch_size)
