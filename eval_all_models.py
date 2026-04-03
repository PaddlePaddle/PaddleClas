import sys
import os
import warnings

import paddle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paddle_utils import *

warnings.filterwarnings("ignore")
import json
import hiera
from PIL import Image
from tqdm import tqdm

MODEL_CONFIGS = {
    "hiera_tiny_224": {"target_acc": 82.8, "embed_dim": 96, "num_heads": 1, "stages": (1, 2, 7, 2)},
    "hiera_small_224": {"target_acc": 83.8, "embed_dim": 96, "num_heads": 1, "stages": (1, 2, 11, 2)},
    "hiera_base_224": {"target_acc": 84.5, "embed_dim": 96, "num_heads": 1, "stages": (2, 3, 16, 3)},
    "hiera_base_plus_224": {"target_acc": 85.2, "embed_dim": 112, "num_heads": 2, "stages": (2, 3, 16, 3)},
    "hiera_large_224": {"target_acc": 86.1, "embed_dim": 144, "num_heads": 2, "stages": (2, 6, 36, 4)},
    "hiera_huge_224": {"target_acc": 86.9, "embed_dim": 256, "num_heads": 4, "stages": (2, 6, 36, 4)},
}

MODEL_URLS = {
    "hiera_tiny_224": "https://dl.fbaipublicfiles.com/hiera/hiera_tiny_224.pth",
    "hiera_small_224": "https://dl.fbaipublicfiles.com/hiera/hiera_small_224.pth",
    "hiera_base_224": "https://dl.fbaipublicfiles.com/hiera/hiera_base_224.pth",
    "hiera_base_plus_224": "https://dl.fbaipublicfiles.com/hiera/hiera_base_plus_224.pth",
    "hiera_large_224": "https://dl.fbaipublicfiles.com/hiera/hiera_large_224.pth",
    "hiera_huge_224": "https://dl.fbaipublicfiles.com/hiera/hiera_huge_224.pth",
}


def download_pytorch_weights(url: str, cache_dir: str = None) -> str:
    import urllib.request
    import time
    
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(__file__), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    filename = url.split("/")[-1]
    cached_file = os.path.join(cache_dir, filename)
    
    if os.path.exists(cached_file):
        file_size = os.path.getsize(cached_file)
        if file_size > 1000000:
            print(f"Using cached file: {cached_file}")
            return cached_file
        else:
            print(f"Cached file incomplete, re-downloading...")
            os.remove(cached_file)
    
    print(f"Downloading: {url}")
    print(f"Saving to: {cached_file}")
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries}...")
            response = urllib.request.urlopen(url, timeout=120)
            total_size = int(response.headers.get('content-length', 0))
            print(f"Total size: {total_size / (1024*1024):.2f} MB")
            
            downloaded = 0
            chunk_size = 8192
            with open(cached_file, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rDownloaded: {downloaded / (1024*1024):.2f} MB ({percent:.1f}%)", end="")
            print("\nDownload complete!")
            return cached_file
        except Exception as e:
            print(f"\nDownload failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in 5 seconds...")
                time.sleep(5)
                if os.path.exists(cached_file):
                    os.remove(cached_file)
            else:
                raise
    
    return cached_file


def convert_pytorch_to_paddle(pytorch_path: str) -> dict:
    import numpy as np
    
    try:
        import torch
    except Exception:
        raise ImportError("PyTorch is required to convert weights. Please install: pip install torch")
    
    print(f"Loading PyTorch weights from: {pytorch_path}")
    
    state_dict = torch.load(pytorch_path, map_location="cpu", weights_only=False)
    
    if isinstance(state_dict, dict):
        if "model_state" in state_dict:
            state_dict = state_dict["model_state"]
            print("Found 'model_state' key, using that.")
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
            print("Found 'state_dict' key, using that.")
        elif "model" in state_dict:
            state_dict = state_dict["model"]
            print("Found 'model' key, using that.")
    
    print(f"Found {len(state_dict)} keys in state_dict")
    
    paddle_state_dict = {}
    converted_keys = []
    
    for key, value in state_dict.items():
        if hasattr(value, "numpy"):
            tensor_value = paddle.to_tensor(value.detach().numpy())
        elif isinstance(value, np.ndarray):
            tensor_value = paddle.to_tensor(value)
        else:
            tensor_value = value
        
        if "mlp.fc" in key and key.endswith(".weight") and len(tensor_value.shape) == 2:
            tensor_value = tensor_value.T
            converted_keys.append(f"{key} (transposed)")
        
        paddle_state_dict[key] = tensor_value
    
    print(f"Converted {len(converted_keys)} MLP weight matrices (transposed)")
    
    return paddle_state_dict


def save_paddle_weights(state_dict: dict, output_path: str):
    print(f"Saving Paddle weights to: {output_path}")
    paddle.save(state_dict, output_path)
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved! File size: {file_size:.2f} MB")


def get_model_with_conversion(model_name, weights_dir="weights", cache_dir="cache"):
    print(f"Loading {model_name}...")
    
    weights_path = os.path.join(weights_dir, f"{model_name}_mae_in1k_ft_in1k.pdparams")
    
    if os.path.exists(weights_path):
        print(f"Loading cached Paddle weights from: {weights_path}")
        config = MODEL_CONFIGS[model_name]
        model = hiera.Hiera(
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            stages=config["stages"],
        )
        state_dict = paddle.load(weights_path)
        model.set_state_dict(state_dict)
        print("Weights loaded successfully!")
        return model
    
    print(f"Paddle weights not found, converting from PyTorch...")
    
    url = MODEL_URLS[model_name]
    pytorch_path = download_pytorch_weights(url, cache_dir)
    
    paddle_state_dict = convert_pytorch_to_paddle(pytorch_path)
    
    os.makedirs(weights_dir, exist_ok=True)
    save_paddle_weights(paddle_state_dict, weights_path)
    
    config = MODEL_CONFIGS[model_name]
    model = hiera.Hiera(
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        stages=config["stages"],
    )
    model.set_state_dict(paddle_state_dict)
    print("Model loaded with converted weights!")
    
    return model


def evaluate_model(model_name, data_dir, synset_to_idx, batch_size=16, device="gpu", num_images=0, weights_dir="weights"):
    print(f"Device: {device}")
    paddle.device.set_device(device)
    
    model = get_model_with_conversion(model_name, weights_dir)
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
    
    class_dirs = sorted(
        [
            d
            for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("n")
        ]
    )
    
    all_files = []
    for class_dir in class_dirs:
        class_path = os.path.join(data_dir, class_dir)
        if class_dir not in synset_to_idx:
            continue
        gt_idx = synset_to_idx[class_dir]
        img_files = [
            (os.path.join(class_path, f), gt_idx)
            for f in os.listdir(class_path)
            if f.lower().endswith((".jpeg", ".jpg", ".png"))
        ]
        all_files.extend(img_files)
    
    if num_images > 0:
        all_files = all_files[:num_images]
        print(f"Limiting to {num_images} images for testing...")
    
    print(f"Total images to evaluate: {len(all_files)}")
    
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    with paddle.no_grad():
        for i in tqdm(range(0, len(all_files), batch_size), desc=f"Evaluating {model_name}"):
            batch_items = all_files[i:i + batch_size]
            images = []
            labels = []
            for img_path, gt_idx in batch_items:
                try:
                    img = Image.open(img_path).convert("RGB")
                    img = transform(img)
                    images.append(img)
                    labels.append(gt_idx)
                except Exception as e:
                    continue
            if not images:
                continue
            images = paddle.stack(images)
            labels_tensor = paddle.to_tensor(labels, dtype="int64")
            outputs = model(images)
            _, top1_pred = outputs._max(1)
            top1_pred = top1_pred.numpy()
            top5_result = outputs.topk(5, axis=1)
            top5_pred = top5_result.indices.numpy()
            labels_np = labels_tensor.numpy()
            total += len(labels_np)
            for idx, label in enumerate(labels_np):
                if top1_pred[idx] == label:
                    correct_top1 += 1
                if label in top5_pred[idx]:
                    correct_top5 += 1
    
    if total == 0:
        return None
    
    top1_acc = 100.0 * correct_top1 / total
    top5_acc = 100.0 * correct_top5 / total
    return {
        "model": model_name,
        "target": target_acc,
        "top1": top1_acc,
        "top5": top5_acc,
        "total": total,
        "gap": top1_acc - target_acc,
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate all Hiera models on ImageNet with auto weight conversion")
    parser.add_argument("--data_dir", type=str, default="imagenet_official/imagenet-val")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="gpu", help="Device to use (gpu/cpu/gpu:0 etc.)")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated list of models to evaluate (default: all)")
    parser.add_argument("--num_images", type=int, default=0, help="Number of images to evaluate (0 = all)")
    parser.add_argument("--weights_dir", type=str, default="weights", help="Directory to save/load Paddle weights")
    args = parser.parse_args()
    
    data_dir = args.data_dir
    batch_size = args.batch_size
    device = args.device
    num_images = args.num_images
    weights_dir = args.weights_dir
    
    print("=" * 60)
    print("Hiera Model Evaluation on ImageNet")
    print("(Auto convert PyTorch weights to Paddle format)")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Data directory: {data_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Num images: {num_images if num_images > 0 else 'all (50000)'}")
    print(f"Weights directory: {weights_dir}")
    
    print("\nLoading synset mapping...")
    mapping_file = "imagenet_synset_to_idx.json"
    if os.path.exists(mapping_file):
        with open(mapping_file, "r") as f:
            synset_to_idx = json.load(f)
        print(f"Loaded {len(synset_to_idx)} synset mappings")
    else:
        class_dirs = sorted(
            [
                d
                for d in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("n")
            ]
        )
        synset_to_idx = {cls_dir: idx for idx, cls_dir in enumerate(class_dirs)}
        with open(mapping_file, "w") as f:
            json.dump(synset_to_idx, f, indent=2)
        print(f"Created {mapping_file}")
    
    if args.models:
        models_to_evaluate = [m.strip() for m in args.models.split(",")]
    else:
        models_to_evaluate = list(MODEL_CONFIGS.keys())
    
    results = []
    for model_name in models_to_evaluate:
        print(f"\n{'=' * 60}")
        print(f"Evaluating {model_name}")
        print(f"{'=' * 60}")
        try:
            result = evaluate_model(model_name, data_dir, synset_to_idx, batch_size, device, num_images, weights_dir)
            if result:
                results.append(result)
                print(
                    f"\n{model_name}: Top-1={result['top1']:.2f}%, Top-5={result['top5']:.2f}%"
                )
                if result['gap'] >= 0:
                    print(f"PASSED: Model meets target accuracy!")
                else:
                    print(f"FAILED: Model is {-result['gap']:.2f}% below target")
            else:
                print(f"\n{model_name}: Failed - no images processed")
        except Exception as e:
            print(f"\n{model_name}: Error - {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(
        f"{'Model':<20} {'Target':>8} {'Top-1':>8} {'Top-5':>8} {'Gap':>8} {'Status':<10}"
    )
    print("-" * 60)
    for r in results:
        status = "PASS" if r["gap"] >= 0 else "FAIL"
        print(
            f"{r['model']:<20} {r['target']:>8.1f}% {r['top1']:>8.2f}% {r['top5']:>8.2f}% {r['gap']:>+8.2f}% {status:<10}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
