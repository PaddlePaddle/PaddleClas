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
import numpy as np
from PIL import Image
from tqdm import tqdm

VIDEO_MODEL_CONFIGS = {
    "hiera_base_16x224": {"target_acc": 84.0, "embed_dim": 96, "num_heads": 1, "stages": (2, 3, 16, 3), "num_classes": 400},
    "hiera_base_plus_16x224": {"target_acc": 85.0, "embed_dim": 112, "num_heads": 2, "stages": (2, 3, 16, 3), "num_classes": 400},
    "hiera_large_16x224": {"target_acc": 87.3, "embed_dim": 144, "num_heads": 2, "stages": (2, 6, 36, 4), "num_classes": 400},
    "hiera_huge_16x224": {"target_acc": 87.8, "embed_dim": 256, "num_heads": 4, "stages": (2, 6, 36, 4), "num_classes": 400},
}

VIDEO_MODEL_URLS = {
    "hiera_base_16x224": "https://dl.fbaipublicfiles.com/hiera/hiera_base_16x224.pth",
    "hiera_base_plus_16x224": "https://dl.fbaipublicfiles.com/hiera/hiera_base_plus_16x224.pth",
    "hiera_large_16x224": "https://dl.fbaipublicfiles.com/hiera/hiera_large_16x224.pth",
    "hiera_huge_16x224": "https://dl.fbaipublicfiles.com/hiera/hiera_huge_16x224.pth",
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


def get_video_model_with_conversion(model_name, weights_dir="weights", cache_dir="cache"):
    print(f"Loading {model_name}...")
    
    weights_path = os.path.join(weights_dir, f"{model_name}_mae_k400_ft_k400.pdparams")
    
    if os.path.exists(weights_path):
        print(f"Loading cached Paddle weights from: {weights_path}")
        config = VIDEO_MODEL_CONFIGS[model_name]
        model = hiera.Hiera(
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            stages=config["stages"],
            num_classes=config["num_classes"],
            input_size=(16, 224, 224),
            q_stride=(1, 2, 2),
            mask_unit_size=(1, 8, 8),
            patch_kernel=(3, 7, 7),
            patch_stride=(2, 4, 4),
            patch_padding=(1, 3, 3),
            sep_pos_embed=True,
        )
        state_dict = paddle.load(weights_path)
        model.set_state_dict(state_dict)
        print("Weights loaded successfully!")
        return model
    
    print(f"Paddle weights not found, converting from PyTorch...")
    
    url = VIDEO_MODEL_URLS[model_name]
    pytorch_path = download_pytorch_weights(url, cache_dir)
    
    paddle_state_dict = convert_pytorch_to_paddle(pytorch_path)
    
    os.makedirs(weights_dir, exist_ok=True)
    save_paddle_weights(paddle_state_dict, weights_path)
    
    config = VIDEO_MODEL_CONFIGS[model_name]
    model = hiera.Hiera(
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        stages=config["stages"],
        num_classes=config["num_classes"],
        input_size=(16, 224, 224),
        q_stride=(1, 2, 2),
        mask_unit_size=(1, 8, 8),
        patch_kernel=(3, 7, 7),
        patch_stride=(2, 4, 4),
        patch_padding=(1, 3, 3),
        sep_pos_embed=True,
    )
    model.set_state_dict(paddle_state_dict)
    print("Model loaded with converted weights!")
    
    return model


def load_video_frames(video_path, num_frames=16, target_size=224):
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV is required for video processing. Install with: pip install opencv-python")
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < num_frames:
        indices = list(range(total_frames))
        while len(indices) < num_frames:
            indices = indices + indices[:num_frames - len(indices)]
        indices = indices[:num_frames]
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frames.append(frame)
        else:
            if frames:
                frames.append(frames[-1])
            else:
                frames.append(Image.new('RGB', (target_size, target_size), (0, 0, 0)))
    
    cap.release()
    return frames


def transform_video_frames(frames, target_size=224):
    transform = paddle.vision.transforms.Compose([
        paddle.vision.transforms.Resize(size=256),
        paddle.vision.transforms.CenterCrop(target_size),
        paddle.vision.transforms.ToTensor(),
        paddle.vision.transforms.Normalize(
            mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]
        ),
    ])
    
    frame_tensors = []
    for frame in frames:
        tensor = transform(frame)
        frame_tensors.append(tensor)
    
    video_tensor = paddle.stack(frame_tensors, axis=1)
    return video_tensor


def evaluate_video_model(model_name, data_dir, label_map, batch_size=8, device="gpu", 
                         num_videos=0, weights_dir="weights"):
    print(f"Device: {device}")
    paddle.device.set_device(device)
    
    model = get_video_model_with_conversion(model_name, weights_dir)
    model.eval()
    target_acc = VIDEO_MODEL_CONFIGS[model_name]["target_acc"]
    
    video_files = []
    for class_name, class_idx in label_map.items():
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for f in os.listdir(class_dir):
                if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                    video_files.append((os.path.join(class_dir, f), class_idx))
    
    if num_videos > 0:
        video_files = video_files[:num_videos]
        print(f"Limiting to {num_videos} videos for testing...")
    
    print(f"Total videos to evaluate: {len(video_files)}")
    
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    with paddle.no_grad():
        for i in tqdm(range(0, len(video_files), batch_size), desc=f"Evaluating {model_name}"):
            batch_items = video_files[i:i + batch_size]
            videos = []
            labels = []
            
            for video_path, gt_idx in batch_items:
                try:
                    frames = load_video_frames(video_path, num_frames=16)
                    video_tensor = transform_video_frames(frames)
                    videos.append(video_tensor)
                    labels.append(gt_idx)
                except Exception as e:
                    print(f"Error loading {video_path}: {e}")
                    continue
            
            if not videos:
                continue
            
            videos = paddle.stack(videos)
            labels_tensor = paddle.to_tensor(labels, dtype="int64")
            
            outputs = model(videos)
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
    
    parser = argparse.ArgumentParser(description="Evaluate Hiera video models on Kinetics-400")
    parser.add_argument("--data_dir", type=str, default="k400_val", help="K400 validation data directory")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="gpu", help="Device to use (gpu/cpu)")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated list of models")
    parser.add_argument("--num_videos", type=int, default=0, help="Number of videos to evaluate (0 = all)")
    parser.add_argument("--weights_dir", type=str, default="weights", help="Directory for weights")
    parser.add_argument("--label_file", type=str, default=None, help="JSON file with class labels")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Hiera Video Model Evaluation on Kinetics-400")
    print("(Auto convert PyTorch weights to Paddle format)")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Data directory: {args.data_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num videos: {args.num_videos if args.num_videos > 0 else 'all'}")
    print(f"Weights directory: {args.weights_dir}")
    
    if args.label_file and os.path.exists(args.label_file):
        with open(args.label_file, "r") as f:
            label_map = json.load(f)
    else:
        if os.path.isdir(args.data_dir):
            class_names = sorted([d for d in os.listdir(args.data_dir) 
                                  if os.path.isdir(os.path.join(args.data_dir, d))])
            label_map = {name: idx for idx, name in enumerate(class_names)}
        else:
            print(f"Data directory not found: {args.data_dir}")
            print("Please provide a valid data directory with K400 validation videos")
            print("Expected structure: data_dir/class_name/video_files.mp4")
            return
    
    print(f"Found {len(label_map)} classes")
    
    if args.models:
        models_to_evaluate = [m.strip() for m in args.models.split(",")]
    else:
        models_to_evaluate = list(VIDEO_MODEL_CONFIGS.keys())
    
    results = []
    for model_name in models_to_evaluate:
        print(f"\n{'=' * 60}")
        print(f"Evaluating {model_name}")
        print(f"{'=' * 60}")
        try:
            result = evaluate_video_model(
                model_name, args.data_dir, label_map, args.batch_size, 
                args.device, args.num_videos, args.weights_dir
            )
            if result:
                results.append(result)
                print(f"\n{model_name}: Top-1={result['top1']:.2f}%, Top-5={result['top5']:.2f}%")
                if result['gap'] >= 0:
                    print(f"PASSED: Model meets target accuracy!")
                else:
                    print(f"FAILED: Model is {-result['gap']:.2f}% below target")
            else:
                print(f"\n{model_name}: Failed - no videos processed")
        except Exception as e:
            print(f"\n{model_name}: Error - {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Model':<25} {'Target':>8} {'Top-1':>8} {'Top-5':>8} {'Gap':>8} {'Status':<10}")
    print("-" * 60)
    for r in results:
        status = "PASS" if r["gap"] >= 0 else "FAIL"
        print(f"{r['model']:<25} {r['target']:>8.1f}% {r['top1']:>8.2f}% {r['top5']:>8.2f}% {r['gap']:>+8.2f}% {status:<10}")
    print("=" * 60)


if __name__ == "__main__":
    main()
