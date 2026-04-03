"""
K400 数据集下载脚本 (Linux 兼容)
只下载验证集 (~20GB)
"""

import os
import subprocess
import sys
import time
import ssl
import urllib.request

BASE_URL = "https://s3.amazonaws.com/kinetics/400"

def download_with_wget(url, output_path, max_retries=5):
    """使用 wget 下载（支持断点续传）"""
    for attempt in range(max_retries):
        try:
            cmd = ["wget", "-c", "-O", output_path, url]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                return True
            print(f"wget 失败 (attempt {attempt+1}/{max_retries})")
        except Exception as e:
            print(f"wget 错误: {e}")
        
        if attempt < max_retries - 1:
            print(f"等待 5 秒后重试...")
            time.sleep(5)
    return False

def download_with_curl(url, output_path, max_retries=5):
    """使用 curl 下载"""
    for attempt in range(max_retries):
        try:
            cmd = ["curl", "-L", "-C", "-", "-o", output_path, url]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                return True
            print(f"curl 失败 (attempt {attempt+1}/{max_retries})")
        except Exception as e:
            print(f"curl 错误: {e}")
        
        if attempt < max_retries - 1:
            print(f"等待 5 秒后重试...")
            time.sleep(5)
    return False

def download_with_python(url, output_path, max_retries=5):
    """使用 Python 下载（带重试）"""
    for attempt in range(max_retries):
        try:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            response = urllib.request.urlopen(url, timeout=120, context=ssl_context)
            total_size = int(response.headers.get('content-length', 0))
            
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
                        print(f"\r  下载进度: {percent:.1f}%", end="")
            
            print()
            return True
        except Exception as e:
            print(f"\n  Python 下载错误: {e}")
            if attempt < max_retries - 1:
                print(f"  等待 5 秒后重试...")
                time.sleep(5)
    
    return False

def download_file(url, output_path, max_retries=5):
    """下载单个文件（自动选择最佳方式）"""
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        if file_size > 1000000:
            print(f"[跳过] 已存在: {os.path.basename(output_path)} ({file_size/(1024*1024):.1f}MB)")
            return True
    
    print(f"[下载] {os.path.basename(output_path)}...")
    
    if sys.platform != "win32":
        if os.system("which wget > /dev/null 2>&1") == 0:
            return download_with_wget(url, output_path, max_retries)
        if os.system("which curl > /dev/null 2>&1") == 0:
            return download_with_curl(url, output_path, max_retries)
    
    return download_with_python(url, output_path, max_retries)

def download_k400_val(output_dir):
    """下载 K400 验证集"""
    val_dir = os.path.join(output_dir, "k400_targz", "val")
    os.makedirs(val_dir, exist_ok=True)
    
    print("=" * 60)
    print("下载 K400 验证集 tar.gz 文件")
    print("=" * 60)
    print(f"输出目录: {val_dir}")
    print()
    
    val_list_url = f"{BASE_URL}/val/k400_val_path.txt"
    val_list_path = os.path.join(output_dir, "k400_val_path.txt")
    
    print(f"获取验证集文件列表...")
    if not download_file(val_list_url, val_list_path):
        print("无法获取文件列表，使用备用方法...")
        urls = get_fallback_urls()
    else:
        with open(val_list_path, "r") as f:
            urls = [line.strip() for line in f if line.strip()]
    
    print(f"找到 {len(urls)} 个文件")
    print(f"预计总大小: ~20GB")
    print()
    
    success = 0
    failed = []
    
    for i, url in enumerate(urls, 1):
        filename = url.split("/")[-1]
        output_path = os.path.join(val_dir, filename)
        print(f"\n[{i}/{len(urls)}] ", end="")
        
        if download_file(url, output_path):
            success += 1
        else:
            failed.append(url)
    
    print()
    print("=" * 60)
    print(f"下载完成! 成功: {success}/{len(urls)}")
    print("=" * 60)
    
    if failed:
        print(f"\n失败的文件 ({len(failed)}):")
        for url in failed:
            print(f"  - {url}")
    
    print(f"\n文件位置: {val_dir}")
    print(f"\n下一步: 解压文件")
    print(f"  python download_k400_val.py --extract -o {output_dir}")

def get_fallback_urls():
    """备用 URL 列表（部分文件）"""
    base = "https://s3.amazonaws.com/kinetics/400/val"
    return [
        f"{base}/k400_val_part_01.tar.gz",
        f"{base}/k400_val_part_02.tar.gz",
        f"{base}/k400_val_part_03.tar.gz",
    ]

def extract_k400_val(output_dir):
    """解压 K400 验证集"""
    import tarfile
    
    val_targz_dir = os.path.join(output_dir, "k400_targz", "val")
    val_dir = os.path.join(output_dir, "k400", "val")
    os.makedirs(val_dir, exist_ok=True)
    
    print("=" * 60)
    print("解压 K400 验证集")
    print("=" * 60)
    
    tar_files = [f for f in os.listdir(val_targz_dir) if f.endswith(".tar.gz")]
    print(f"找到 {len(tar_files)} 个 tar.gz 文件")
    
    for i, tar_name in enumerate(sorted(tar_files), 1):
        tar_path = os.path.join(val_targz_dir, tar_name)
        print(f"[{i}/{len(tar_files)}] 解压 {tar_name}...")
        
        try:
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(val_dir)
        except Exception as e:
            print(f"  错误: {e}")
    
    print()
    print("=" * 60)
    print("解压完成!")
    print("=" * 60)
    print(f"视频位置: {val_dir}")

def download_annotations(output_dir):
    """下载标注文件"""
    ann_dir = os.path.join(output_dir, "k400", "annotations")
    os.makedirs(ann_dir, exist_ok=True)
    
    print("下载标注文件...")
    
    urls = {
        "train.csv": f"{BASE_URL}/annotations/train.csv",
        "val.csv": f"{BASE_URL}/annotations/val.csv",
        "test.csv": f"{BASE_URL}/annotations/test.csv",
    }
    
    for filename, url in urls.items():
        output_path = os.path.join(ann_dir, filename)
        download_file(url, output_path)
    
    print(f"标注文件位置: {ann_dir}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="K400 验证集下载工具")
    parser.add_argument("--output", "-o", type=str, default="./k400_data",
                        help="输出目录 (默认: ./k400_data)")
    parser.add_argument("--download", "-d", action="store_true",
                        help="下载 tar.gz 文件")
    parser.add_argument("--extract", "-e", action="store_true",
                        help="解压 tar.gz 文件")
    parser.add_argument("--annotations", "-a", action="store_true",
                        help="下载标注文件")
    parser.add_argument("--all", action="store_true",
                        help="下载并解压全部")
    
    args = parser.parse_args()
    
    if args.all:
        download_k400_val(args.output)
        extract_k400_val(args.output)
        download_annotations(args.output)
    elif args.download:
        download_k400_val(args.output)
    elif args.extract:
        extract_k400_val(args.output)
    elif args.annotations:
        download_annotations(args.output)
    else:
        print("=" * 60)
        print("K400 验证集下载工具")
        print("=" * 60)
        print(f"""
使用方法:

# 下载验证集 tar.gz 文件 (~20GB)
python download_k400_val.py --download -o {args.output}

# 解压文件
python download_k400_val.py --extract -o {args.output}

# 下载标注文件
python download_k400_val.py --annotations -o {args.output}

# 全部执行
python download_k400_val.py --all -o {args.output}
""")

if __name__ == "__main__":
    main()
