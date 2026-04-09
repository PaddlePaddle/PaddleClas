"""
下载并转换所有视频模型权重
"""
import torch
import numpy as np
import paddle
import os
import urllib.request
import ssl

VIDEO_MODEL_URLS = {
    "hiera_base_16x224": "https://dl.fbaipublicfiles.com/hiera/hiera_base_16x224.pth",
    "hiera_base_plus_16x224": "https://dl.fbaipublicfiles.com/hiera/hiera_base_plus_16x224.pth",
    "hiera_large_16x224": "https://dl.fbaipublicfiles.com/hiera/hiera_large_16x224.pth",
    "hiera_huge_16x224": "https://dl.fbaipublicfiles.com/hiera/hiera_huge_16x224.pth",
}

def download_file(url, output_path):
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        if file_size > 100000000:
            print(f"Already exists: {output_path} ({file_size / (1024*1024):.2f} MB)")
            return True
        else:
            print(f"File incomplete, re-downloading...")
            os.remove(output_path)
    
    print(f"Downloading: {url}")
    
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    response = urllib.request.urlopen(url, timeout=300, context=ssl_context)
    total_size = int(response.headers.get('content-length', 0))
    print(f"Total size: {total_size / (1024*1024):.2f} MB")
    
    downloaded = 0
    chunk_size = 8192
    with open(output_path, 'wb') as f:
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
    return True

def convert_weights(pytorch_path, output_path):
    if os.path.exists(output_path):
        print(f"Already converted: {output_path}")
        return True
    
    print(f"Loading PyTorch weights from: {pytorch_path}")
    
    state_dict = torch.load(pytorch_path, map_location="cpu", weights_only=False)
    
    if isinstance(state_dict, dict):
        if "model_state" in state_dict:
            state_dict = state_dict["model_state"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]
    
    print(f"Found {len(state_dict)} keys")
    
    paddle_state_dict = {}
    converted = 0
    
    for key, value in state_dict.items():
        if hasattr(value, "numpy"):
            tensor_value = paddle.to_tensor(value.detach().numpy())
        elif isinstance(value, np.ndarray):
            tensor_value = paddle.to_tensor(value)
        else:
            tensor_value = value
        
        if "mlp.fc" in key and key.endswith(".weight") and len(tensor_value.shape) == 2:
            tensor_value = tensor_value.T
            converted += 1
        
        paddle_state_dict[key] = tensor_value
    
    print(f"Converted {converted} MLP weights")
    
    print(f"Saving to: {output_path}")
    paddle.save(paddle_state_dict, output_path)
    print(f"Done! Size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    return True

def main():
    os.makedirs("cache", exist_ok=True)
    os.makedirs("weights", exist_ok=True)
    
    print("=" * 60)
    print("Downloading and converting all video model weights")
    print("=" * 60)
    
    for model_name, url in VIDEO_MODEL_URLS.items():
        print(f"\n{'=' * 60}")
        print(f"Processing: {model_name}")
        print(f"{'=' * 60}")
        
        pytorch_path = os.path.join("cache", f"{model_name}.pth")
        paddle_path = os.path.join("weights", f"{model_name}_mae_k400_ft_k400.pdparams")
        
        try:
            download_file(url, pytorch_path)
            convert_weights(pytorch_path, paddle_path)
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("All models processed!")
    print("=" * 60)
    
    print("\nWeights in weights/:")
    for f in os.listdir("weights"):
        if f.endswith(".pdparams"):
            path = os.path.join("weights", f)
            size = os.path.getsize(path) / (1024*1024)
            print(f"  {f}: {size:.2f} MB")

if __name__ == "__main__":
    main()
