import sys
import os
import warnings
import paddle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paddle_utils import *

warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image
from tqdm import tqdm
import json

import hiera

def load_model_configs():
    config_path = os.path.join(os.path.dirname(__file__), "model_configs.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {
        "image": {
            "hiera_tiny_224": {"target_acc": 82.8},
            "hiera_small_224": {"target_acc": 83.8},
            "hiera_base_224": {"target_acc": 84.5},
            "hiera_base_plus_224": {"target_acc": 85.2},
            "hiera_large_224": {"target_acc": 86.1},
            "hiera_huge_224": {"target_acc": 86.9},
        },
        "video": {
            "hiera_base_16x224": {"target_acc": 84.0},
            "hiera_base_plus_16x224": {"target_acc": 85.0},
            "hiera_large_16x224": {"target_acc": 87.3},
            "hiera_huge_16x224": {"target_acc": 87.8},
        }
    }

def load_k400_classes():
    classes_file = os.path.join(os.path.dirname(__file__), "k400_classes.txt")
    if os.path.exists(classes_file):
        with open(classes_file, "r") as f:
            content = f.read()
            import re
            classes = re.findall(r'"([^"]+)"', content)
            if classes:
                print(f"Loaded {len(classes)} K400 classes from config file")
                return classes
    return [
        "abseiling", "air drumming", "answering questions", "applauding", "applying cream",
        "archery", "arm wrestling", "arranging flowers", "assembling computer", "auctioning",
        "baby waking up", "baking cookies", "balloon blowing", "bandaging", "barbequing",
        "bartending", "beatboxing", "bee keeping", "belly dancing", "bench pressing",
        "bending back", "bending metal", "biking through snow", "blasting sand", "blowing glass",
        "blowing leaves", "blowing nose", "blowing out candles", "bobsledding", "bookbinding",
        "bouncing on trampoline", "bowling", "braiding hair", "breading or breadcrumbing", "breakdancing",
        "brush painting", "brushing hair", "brushing teeth", "building cabinet", "building shed",
        "bungee jumping", "busking", "canoeing or kayaking", "capoeira", "carrying baby",
        "cartwheeling", "carving pumpkin", "catching fish", "catching or throwing baseball", "catching or throwing frisbee",
        "catching or throwing softball", "celebrating", "changing oil", "changing wheel", "checking tires",
        "cheerleading", "chopping wood", "clapping", "clay pottery making", "clean and jerk",
        "cleaning floor", "cleaning gutters", "cleaning pool", "cleaning shoes", "cleaning toilet",
        "cleaning windows", "climbing a rope", "climbing ladder", "climbing tree", "contact juggling",
        "cooking chicken", "cooking egg", "cooking on campfire", "cooking sausages", "counting money",
        "country line dancing", "cracking neck", "crawling baby", "crossing river", "crying",
        "curling hair", "cutting nails", "cutting pineapple", "cutting watermelon", "dancing ballet",
        "dancing charleston", "dancing gangnam style", "dancing macarena", "deadlifting", "decorating the christmas tree",
        "digging", "dining", "disc golfing", "diving cliff", "dodgeball",
        "doing aerobics", "doing laundry", "doing nails", "drawing", "dribbling basketball",
        "drinking", "drinking beer", "drinking shots", "driving car", "driving tractor",
        "drop kicking", "drumming fingers", "dunking basketball", "dying hair", "eating burger",
        "eating cake", "eating carrots", "eating chips", "eating doughnuts", "eating hotdog",
        "eating ice cream", "eating spaghetti", "eating watermelon", "egg hunting", "exercising arm",
        "exercising with an exercise ball", "extinguishing fire", "faceplanting", "feeding birds", "feeding fish",
        "feeding goats", "filling eyebrows", "finger snapping", "fixing hair", "flipping pancake",
        "flying kite", "folding clothes", "folding napkins", "folding paper", "front raises",
        "frying vegetables", "garbage collecting", "gargling", "getting a haircut", "getting a tattoo",
        "giving or receiving award", "golf chipping", "golf driving", "golf putting", "grinding meat",
        "grooming dog", "grooming horse", "gymnastics tumbling", "hammer throw", "headbanging",
        "headbutting", "high jump", "high kick", "hitting baseball", "hockey stop",
        "holding snake", "hopscotch", "hoverboarding", "hugging", "hula hooping",
        "hurdling", "hurling (sport)", "ice climbing", "ice fishing", "ice skating",
        "ironing", "javelin throw", "jetskiing", "jogging", "juggling balls",
        "juggling fire", "juggling soccer ball", "jumping into pool", "jumpstyle dancing", "kicking field goal",
        "kicking soccer ball", "kissing", "kitesurfing", "knitting", "krumping",
        "laughing", "laying bricks", "long jump", "lunge", "making a cake",
        "making a sandwich", "making bed", "making jewelry", "making pizza", "making snowman",
        "making sushi", "making tea", "marching", "massaging back", "massaging feet",
        "massaging legs", "massaging person's head", "milking cow", "mopping floor", "motorcycling",
        "moving furniture", "mowing lawn", "news anchoring", "opening bottle", "opening present",
        "paragliding", "parasailing", "parkour", "passing American football (in game)", "passing American football (not in game)",
        "peeling apples", "peeling potatoes", "petting animal (not cat)", "petting cat", "picking fruit",
        "planting trees", "plastering", "playing accordion", "playing badminton", "playing bagpipes",
        "playing basketball", "playing bass guitar", "playing cards", "playing cello", "playing chess",
        "playing clarinet", "playing controller", "playing cricket", "playing cymbals", "playing didgeridoo",
        "playing drums", "playing flute", "playing guitar", "playing harmonica", "playing harp",
        "playing keyboard", "playing kickball", "playing lute", "playing monopoly", "playing organ",
        "playing paintball", "playing piano", "playing poker", "playing recorder", "playing rugby",
        "playing saxophone", "playing scrabble", "playing snooker", "playing soccer", "playing softball",
        "playing squash", "playing tennis", "playing trombone", "playing trumpet", "playing ukulele",
        "playing violin", "playing volleyball", "playing with trains", "playing xylophone", "poking bellybutton",
        "pole vault", "popping balloons", "pouring beer", "pouring coffee", "pouring milk",
        "pouring wine", "praying", "pregnancy test", "preparing salad", "presenting weather forecast",
        "pressing buttons", "proposing", "punching bag", "punching person (boxing)", "push up",
        "pushing car", "pushing cart", "pushing wheelchair", "reading book", "reading newspaper",
        "recording music", "riding a bike", "riding camel", "riding elephant", "riding mechanical bull",
        "riding mountain bike", "riding mule", "riding or walking with horse", "riding scooter", "riding unicycle",
        "ripping paper", "roasting marshmallows", "roasting pig", "robot dancing", "rock climbing",
        "rock scissors paper", "rolling pastry", "rolling tire", "romance", "rope climbing",
        "rowing", "rugby tackle", "running on treadmill", "salsa dancing", "sanding floor",
        "sawing wood", "scrambling eggs", "scuba diving", "setting table", "sewing",
        "shaking hands", "shaking head", "sharpening knives", "sharpening pencil", "shaving head",
        "shaving legs", "shearing sheep", "shining shoes", "shooting basketball", "shooting goal (soccer)",
        "shooting gun", "shopping", "shot put", "shoveling snow", "shredding paper",
        "shuffling cards", "shuffling feet", "singing", "sitting in chair", "situp",
        "skateboarding", "ski jumping", "skiing (not slalom or crosscountry)", "skiing crosscountry", "skiing slalom",
        "skipping rope", "skydiving", "slacklining", "slapping", "sliding door",
        "smoking", "smoking hookah", "snatch weight lifting", "sneezing", "sniffing",
        "snorkeling", "snowboarding", "snowkiting", "snowmobiling", "somersaulting",
        "spinning poi", "spray painting", "spraying", "sprinting", "squat",
        "stacking cups", "stacking shelves", "sticking tongue out", "stomach crunches", "stretching arm",
        "stretching leg", "strumming guitar", "surfing crowd", "surfing water", "sweeping floor",
        "swimming backstroke", "swimming breast stroke", "swimming butterfly stroke", "swimming front crawl", "swimming underwater",
        "swinging baseball", "swinging on something", "sword fighting", "syringing", "table soccer",
        "tai chi", "taking a photo", "talking on cell phone", "tango dancing", "tap dancing",
        "tapping guitar", "tapping pen", "tasting beer", "tasting food", "tasting wine",
        "texting", "threading eyebrow", "throwing axe", "throwing ball", "throwing discus",
        "throwing knife", "throwing snowball", "throwing water balloon", "throwing (sport)", "tickling",
        "tie tying", "tightrope walking", "tiptoeing", "tobogganing", "tossing coin",
        "tossing salad", "towing", "training dog", "trapezing", "trimming or shaving beard",
        "trimming trees", "triple jump", "tying tie", "unboxing", "unloading truck",
        "using computer", "using remote controller (not gaming)", "using segway", "vacuuming floor", "visiting zoo",
        "waking up", "walking the dog", "walking through snow", "walking with crutches", "wall pushups",
        "washing dishes", "washing face", "washing hair", "washing hands", "watching tv",
        "water skiing", "water sliding", "watering plants", "waxing back", "waxing chest",
        "waxing eyebrows", "waxing legs", "weaving basket", "weaving fabric", "welding",
        "whistling", "windsurfing", "wrapping present", "wrestling", "writing",
        "yawning", "yoga", "zumba",
    ]

def load_synset_mapping():
    mapping_file = os.path.join(os.path.dirname(__file__), "imagenet_synset_to_idx.json")
    if os.path.exists(mapping_file):
        with open(mapping_file, "r") as f:
            synset_to_idx = json.load(f)
        print(f"Loaded {len(synset_to_idx)} synset mappings from config file")
        return synset_to_idx
    return None

MODEL_CONFIGS = load_model_configs()
IMAGE_MODEL_CONFIGS = MODEL_CONFIGS.get("image", {})
VIDEO_MODEL_CONFIGS = MODEL_CONFIGS.get("video", {})
K400_CLASSES = load_k400_classes()
SYNSET_TO_IDX = load_synset_mapping()

def get_model(model_name):
    print(f"Loading {model_name}...")
    
    model_func = getattr(hiera, model_name, None)
    if model_func is None:
        raise ValueError(f"Unknown model: {model_name}")
    
    if "16x224" in model_name:
        model = model_func(pretrained=True, checkpoint="mae_k400_ft_k400")
    else:
        model = model_func(pretrained=True, checkpoint="mae_in1k_ft_in1k")
    
    return model

def get_synset_mapping(data_dir):
    if SYNSET_TO_IDX:
        return SYNSET_TO_IDX
    
    if not os.path.exists(data_dir):
        return None
    
    class_dirs = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("n")
    ])
    synset_to_idx = {cls_dir: idx for idx, cls_dir in enumerate(class_dirs)}
    
    mapping_file = os.path.join(os.path.dirname(__file__), "imagenet_synset_to_idx.json")
    with open(mapping_file, "w") as f:
        json.dump(synset_to_idx, f, indent=2)
    print(f"Created synset mapping with {len(synset_to_idx)} classes")
    return synset_to_idx

def evaluate_image_model(model_name, data_dir, batch_size, device, num_images):
    print(f"Device: {device}")
    if device != "cpu":
        paddle.device.set_device(device)
    
    model = get_model(model_name)
    model.eval()
    target_acc = IMAGE_MODEL_CONFIGS[model_name]["target_acc"]
    
    synset_to_idx = get_synset_mapping(data_dir)
    if not synset_to_idx:
        print(f"No synset mapping found")
        return None
    
    transform = paddle.vision.transforms.Compose([
        paddle.vision.transforms.Resize(256),
        paddle.vision.transforms.CenterCrop(224),
        paddle.vision.transforms.ToTensor(),
        paddle.vision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    class_dirs = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("n")
    ])
    
    image_files = []
    for class_dir in class_dirs:
        if class_dir not in synset_to_idx:
            continue
        gt_idx = synset_to_idx[class_dir]
        class_path = os.path.join(data_dir, class_dir)
        files = [
            (os.path.join(class_path, f), gt_idx)
            for f in os.listdir(class_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        image_files.extend(files)
    
    if num_images > 0:
        image_files = image_files[:num_images]
        print(f"Limiting to {num_images} images for testing...")
    
    if not image_files:
        print(f"No images found in {data_dir}")
        return None
    
    print(f"Total images: {len(image_files)}")
    
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    with paddle.no_grad():
        for i in tqdm(range(0, len(image_files), batch_size), desc=f"Evaluating {model_name}"):
            batch_files = image_files[i:i+batch_size]
            images = []
            labels_batch = []
            
            for img_path, gt_idx in batch_files:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = transform(img)
                    images.append(img_tensor)
                    labels_batch.append(gt_idx)
                except Exception as e:
                    continue
            
            if not images:
                continue
            
            images = paddle.stack(images)
            outputs = model(images)
            
            _, top1_pred = outputs._max(1)
            top1_pred = top1_pred.numpy()
            top5 = outputs.topk(5, axis=1)
            top5_pred = top5.indices.numpy()
            
            total += len(labels_batch)
            for idx, label in enumerate(labels_batch):
                if top1_pred[idx] == label:
                    correct_top1 += 1
                if label in top5_pred[idx]:
                    correct_top5 += 1
    
    if total == 0:
        return None
    
    return {
        "model": model_name,
        "type": "image",
        "top1": 100.0 * correct_top1 / total,
        "top5": 100.0 * correct_top5 / total,
        "total": total,
        "target": target_acc
    }

def load_video_frames(video_path, num_frames=16):
    import cv2
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames >= num_frames:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        indices = list(range(total_frames))
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(frame))
        else:
            if frames:
                frames.append(frames[-1])
            else:
                frames.append(Image.new('RGB', (224, 224), (0, 0, 0)))
    
    cap.release()
    return frames

def transform_video_frames(frames):
    transform = paddle.vision.transforms.Compose([
        paddle.vision.transforms.Resize(256),
        paddle.vision.transforms.CenterCrop(224),
        paddle.vision.transforms.ToTensor(),
        paddle.vision.transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
    ])
    
    tensors = [transform(f) for f in frames]
    video_tensor = paddle.stack(tensors, axis=1)
    return video_tensor

def evaluate_video_model(model_name, data_dir, batch_size, device, num_videos):
    print(f"Device: {device}")
    if device != "cpu":
        paddle.device.set_device(device)
    
    model = get_model(model_name)
    model.eval()
    target_acc = VIDEO_MODEL_CONFIGS[model_name]["target_acc"]
    
    video_files = []
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            for f in os.listdir(class_path):
                if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_files.append((os.path.join(class_path, f), class_name))
    
    if num_videos > 0:
        video_files = video_files[:num_videos]
        print(f"Limiting to {num_videos} videos for testing...")
    
    if not video_files:
        print(f"No videos found in {data_dir}")
        return None
    
    print(f"Total videos: {len(video_files)}")
    
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    for i in tqdm(range(0, len(video_files), batch_size), desc=f"Evaluating {model_name}"):
        batch_files = video_files[i:i+batch_size]
        videos = []
        class_names = []
        
        for video_path, class_name in batch_files:
            try:
                frames = load_video_frames(video_path)
                video_tensor = transform_video_frames(frames)
                videos.append(video_tensor)
                class_names.append(class_name)
            except Exception as e:
                print(f"Error loading {video_path}: {e}")
                continue
        
        if not videos:
            continue
        
        videos = paddle.stack(videos)
        
        with paddle.no_grad():
            outputs = model(videos)
        
        top5 = outputs.topk(5, axis=1)
        top5_pred = top5.indices.numpy()
        
        total += len(class_names)
        for idx, class_name in enumerate(class_names):
            class_name_normalized = class_name.replace("_", " ")
            top5_classes = [K400_CLASSES[i] if i < len(K400_CLASSES) else f"class_{i}" 
                            for i in top5_pred[idx]]
            
            if class_name_normalized in top5_classes[:1]:
                correct_top1 += 1
            if class_name_normalized in top5_classes:
                correct_top5 += 1
    
    if total == 0:
        return None
    
    return {
        "model": model_name,
        "type": "video",
        "top1": 100.0 * correct_top1 / total,
        "top5": 100.0 * correct_top5 / total,
        "total": total,
        "target": target_acc
    }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate all Hiera models (image + video)")
    parser.add_argument("--image_data_dir", type=str, default="imagenet_official/imagenet-val",
                        help="ImageNet validation data directory")
    parser.add_argument("--video_data_dir", type=str, default="./video_test_data/train",
                        help="Video test data directory")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="gpu", help="Device (gpu/cpu)")
    parser.add_argument("--models", type=str, default="all",
                        help="Models to evaluate: 'all', 'image', 'video', or comma-separated list")
    parser.add_argument("--num_images", type=int, default=0, 
                        help="Number of images to evaluate (0 = all)")
    parser.add_argument("--num_videos", type=int, default=0,
                        help="Number of videos to evaluate (0 = all)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("Hiera Unified Model Evaluation")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    
    if args.models == "all":
        image_models = list(IMAGE_MODEL_CONFIGS.keys())
        video_models = list(VIDEO_MODEL_CONFIGS.keys())
    elif args.models == "image":
        image_models = list(IMAGE_MODEL_CONFIGS.keys())
        video_models = []
    elif args.models == "video":
        image_models = []
        video_models = list(VIDEO_MODEL_CONFIGS.keys())
    else:
        selected = [m.strip() for m in args.models.split(",")]
        image_models = [m for m in selected if m in IMAGE_MODEL_CONFIGS]
        video_models = [m for m in selected if m in VIDEO_MODEL_CONFIGS]
    
    print(f"Image models: {image_models if image_models else 'None'}")
    print(f"Video models: {video_models if video_models else 'None'}")
    
    image_results = []
    video_results = []
    
    for model_name in image_models:
        print(f"\n{'=' * 70}")
        print(f"Evaluating {model_name}")
        print("=" * 70)
        try:
            result = evaluate_image_model(
                model_name, args.image_data_dir, args.batch_size,
                args.device, args.num_images
            )
            if result:
                image_results.append(result)
                print(f"\n{model_name}: Top-1={result['top1']:.2f}%, Top-5={result['top5']:.2f}%")
        except Exception as e:
            print(f"\n{model_name}: Error - {e}")
    
    for model_name in video_models:
        print(f"\n{'=' * 70}")
        print(f"Evaluating {model_name}")
        print("=" * 70)
        try:
            result = evaluate_video_model(
                model_name, args.video_data_dir, args.batch_size,
                args.device, args.num_videos
            )
            if result:
                video_results.append(result)
                print(f"\n{model_name}: Top-1={result['top1']:.2f}%, Top-5={result['top5']:.2f}%")
        except Exception as e:
            print(f"\n{model_name}: Error - {e}")
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    
    if image_results:
        print("\n--- Image Models (ImageNet) ---")
        print(f"{'Model':<25} {'Top-1':>8} {'Top-5':>8} {'Target':>8} {'Images':>8}")
        print("-" * 70)
        for r in image_results:
            print(f"{r['model']:<25} {r['top1']:>8.2f}% {r['top5']:>8.2f}% {r['target']:>8.1f}% {r['total']:>8}")
    
    if video_results:
        print("\n--- Video Models (Kinetics-400) ---")
        print(f"{'Model':<25} {'Top-1':>8} {'Top-5':>8} {'Target':>8} {'Videos':>8}")
        print("-" * 70)
        for r in video_results:
            print(f"{r['model']:<25} {r['top1']:>8.2f}% {r['top5']:>8.2f}% {r['target']:>8.1f}% {r['total']:>8}")
    
    print("=" * 70)

if __name__ == "__main__":
    main()
