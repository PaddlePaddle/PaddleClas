import os
import argparse
import numpy as np
import torch
import paddle


MODEL_URLS = {
    "hiera_tiny_224": {
        "mae_in1k_ft_in1k": "https://dl.fbaipublicfiles.com/hiera/hiera_tiny_224.pth",
        "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_tiny_224.pth",
    },
    "hiera_small_224": {
        "mae_in1k_ft_in1k": "https://dl.fbaipublicfiles.com/hiera/hiera_small_224.pth",
        "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_small_224.pth",
    },
    "hiera_base_224": {
        "mae_in1k_ft_in1k": "https://dl.fbaipublicfiles.com/hiera/hiera_base_224.pth",
        "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_224.pth",
    },
    "hiera_base_plus_224": {
        "mae_in1k_ft_in1k": "https://dl.fbaipublicfiles.com/hiera/hiera_base_plus_224.pth",
        "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_plus_224.pth",
    },
    "hiera_large_224": {
        "mae_in1k_ft_in1k": "https://dl.fbaipublicfiles.com/hiera/hiera_large_224.pth",
        "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_large_224.pth",
    },
    "hiera_huge_224": {
        "mae_in1k_ft_in1k": "https://dl.fbaipublicfiles.com/hiera/hiera_huge_224.pth",
        "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_huge_224.pth",
    },
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
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries}...")
            response = urllib.request.urlopen(url, timeout=60)
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


def convert_model(model_name: str, checkpoint: str, output_dir: str = None, cache_dir: str = None):
    if model_name not in MODEL_URLS:
        print(f"Error: Unknown model '{model_name}'")
        print(f"Available models: {list(MODEL_URLS.keys())}")
        return None
    
    checkpoints = MODEL_URLS[model_name]
    if checkpoint not in checkpoints:
        print(f"Error: Unknown checkpoint '{checkpoint}' for model '{model_name}'")
        print(f"Available checkpoints: {list(checkpoints.keys())}")
        return None
    
    url = checkpoints[checkpoint]
    
    pytorch_path = download_pytorch_weights(url, cache_dir)
    
    paddle_state_dict = convert_pytorch_to_paddle(pytorch_path)
    
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "weights")
    os.makedirs(output_dir, exist_ok=True)
    
    output_filename = f"{model_name}_{checkpoint}.pdparams"
    output_path = os.path.join(output_dir, output_filename)
    
    save_paddle_weights(paddle_state_dict, output_path)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert Hiera PyTorch weights to Paddle format")
    parser.add_argument("--model", type=str, default=None, 
                        help="Model name to convert (default: all models)")
    parser.add_argument("--checkpoint", type=str, default="mae_in1k_ft_in1k",
                        help="Checkpoint type (default: mae_in1k_ft_in1k)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for .pdparams files")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Cache directory for downloaded PyTorch weights")
    parser.add_argument("--list", action="store_true",
                        help="List available models and checkpoints")
    
    args = parser.parse_args()
    
    if args.list:
        print("Available models and checkpoints:")
        print("=" * 60)
        for model_name, checkpoints in MODEL_URLS.items():
            print(f"\n{model_name}:")
            for ckpt_name, url in checkpoints.items():
                print(f"  - {ckpt_name}: {url}")
        return
    
    if args.model:
        convert_model(args.model, args.checkpoint, args.output_dir, args.cache_dir)
    else:
        print("Converting all models...")
        print("=" * 60)
        for model_name in MODEL_URLS.keys():
            print(f"\n{'=' * 60}")
            print(f"Converting: {model_name}")
            print("=" * 60)
            convert_model(model_name, args.checkpoint, args.output_dir, args.cache_dir)
        
        print("\n" + "=" * 60)
        print("All conversions complete!")
        print("=" * 60)


if __name__ == "__main__":
    main()
