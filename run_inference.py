import sys

import paddle

sys.path.append(".")
import hiera
import matplotlib.pyplot as plt
from PIL import Image

print("=" * 60)
print("🚀 Hiera Inference 自动运行脚本")
print("=" * 60)
print("\n📸 第一部分：图像推理")
print("-" * 60)
print("🔄 加载 Hiera-Base-224 模型...")
model = hiera.hiera_base_224(pretrained=True, checkpoint="mae_in1k_ft_in1k")
model.eval()
print("✅ 模型加载完成！")
input_size = 224
transform_list = [
    paddle.vision.transforms.Resize(
        size=int(256 / 224 * input_size),
        interpolation="bicubic",
    ),
    paddle.vision.transforms.CenterCrop(input_size),
]
transform_norm = paddle.vision.transforms.Compose(
    transform_list
    + [
        paddle.vision.transforms.ToTensor(),
        paddle.vision.transforms.Normalize(
            mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
        ),
    ]
)
img_path = "/media/user/40da1a25-a924-43a2-b274-e33d39ea8680/cv/lzy/hiera-main/hiera-main/examples/img/dog.jpg"
print(f"🖼️  加载图像: {img_path}")
img = Image.open(img_path).convert("RGB")
img_norm = transform_norm(img)
print("🔄 执行推理...")
with paddle.no_grad():
    output = model(img_norm[None, ...])
pred_idx = output.argmax(dim=-1).item()
print(f"🎯 预测类别索引: {pred_idx}")
imagenet_labels = {
    (207): "golden retriever (金毛犬)",
    (243): "bull mastiff (斗牛獒)",
    (244): "chow chow (松狮犬)",
    (245): "keeshond (荷兰毛狮犬)",
    (246): "Great Dane (大丹犬)",
    (247): "German shepherd (德国牧羊犬)",
    (248): "collie (柯利牧羊犬)",
    (249): "Border collie (边境牧羊犬)",
    (250): "Shetland sheepdog (设得兰牧羊犬)",
    (251): "Scotch terrier (苏格兰梗)",
    (252): "cairn terrier (凯恩梗)",
    (253): "Airedale terrier (艾尔谷梗)",
    (254): "Australian terrier (澳大利亚梗)",
    (255): "Dandie Dinmont terrier (丹迪丁蒙梗)",
    (256): "Boston bull (波士顿梗)",
}
pred_label = imagenet_labels.get(pred_idx, f"类别 {pred_idx}")
print(f"🗂️  预测类别名称: {pred_label}")
print("\n📊 获取中间特征图...")
with paddle.no_grad():
    _, intermediates = model(img_norm[None, ...], return_intermediates=True)
print("中间特征图形状:")
for i, feat in enumerate(intermediates):
    print(f"  阶段 {i + 1}: {feat.shape}")
print("\n" + "=" * 60)
print("🎬 第二部分：视频推理")
print("-" * 60)
print("🔄 加载 Hiera-Base-16x224 视频模型...")
video_model = hiera.hiera_base_16x224(pretrained=True, checkpoint="mae_k400_ft_k400")
video_model.eval()
print("✅ 视频模型加载完成！")
vid_path = "/media/user/40da1a25-a924-43a2-b274-e33d39ea8680/cv/lzy/hiera-main/hiera-main/examples/vid/dog.mp4"
print(f"🎥 加载视频: {vid_path}")
try:
    frames, audio, info = torchvision.io.read_video(
        vid_path, pts_unit="sec", output_format="THWC"
    )
    frames = frames.float() / 255
    print(f"✅ 视频加载完成！FPS: {info['video_fps']}, 总帧数: {len(frames)}")
    print("🔄 采样帧...")
    frames = paddle.stack([frames[:64], frames[64:128]], dim=0)
    frames = frames[:, ::4]
    print(f"采样后形状: {frames.shape}")
    frames = frames.permute(0, 4, 1, 2, 3).contiguous()
    frames = paddle.nn.functional.interpolate(
        frames, size=(16, 224, 224), mode="trilinear"
    )
    print(f"调整后形状: {frames.shape}")
    frames = frames - paddle.tensor([0.45, 0.45, 0.45]).view(1, -1, 1, 1, 1)
    frames = frames / paddle.tensor([0.225, 0.225, 0.225]).view(1, -1, 1, 1, 1)
    print("🔄 执行视频推理...")
    with paddle.no_grad():
        out = video_model(frames)
        out = out.mean(0)
    pred_idx = out.argmax(dim=-1).item()
    print(f"🎯 视频预测类别索引: {pred_idx}")
    kinetics_labels = {
        (0): "abseiling (速降)",
        (1): "air drumming (空气鼓)",
        (2): "answering questions (回答问题)",
        (3): "applauding (鼓掌)",
        (4): "applying cream (涂抹面霜)",
        (5): "archery (射箭)",
        (6): "arm wrestling (掰手腕)",
        (7): "arranging flowers (插花)",
        (8): "assembling computer (组装电脑)",
        (9): "auctioning (拍卖)",
        (363): "training dog (训练狗)",
        (125): "feeding goat (喂山羊)",
    }
    pred_label = kinetics_labels.get(pred_idx, f"类别 {pred_idx}")
    print(f"🗂️  视频预测类别名称: {pred_label}")
except Exception as e:
    print(f"⚠️ 视频推理出错: {e}")
    print("提示: 请确保已安装 PyAV (pip install av)")
print("\n" + "=" * 60)
print("🎉 Inference 脚本运行完成！")
print("=" * 60)
