# MobileNetV5-base CINN 复现

单卡完整分类模型、FP32、关闭 TF32，无 AMP；固定随机初始权重。两端加载相同权重与数据，训练 AdamW lr=1e-5、weight_decay=0.01，batch=16、输入224×224；预处理为像素除以255。原生 BatchNorm 保留，其余使用默认 CINN 分解，不额外保留 Linear。

## 环境

初始权重及断点恢复均通过共享 `strict_set_state_dict` 检查键集合、张量形状和加载返回值；不兼容时明确报错，不依赖 `assert`。校验单测可在仓库根目录执行：`python -O -m unittest discover -s tests/cinn/_shared -p test_checkpoint_utils.py -v`。

两端均显式设置 drop_rate=0.0，与历史对齐实验一致，不把随机 dropout 差异计入 CINN 比较。

本地验证环境：V100S 32GB、Python3.10.20、Paddle3.5.0.dev20260722（08de6be508389d5277b02e78fbc8742f1ed78160，编译CUDA12.6）、numpy1.26.4；安装对应 Paddle GPU 包和 PaddleClas requirements。


## 数据与权重

data/data.npz 为历史 ImageNet100 子集的前320张训练和16张验证图片，已预处理成RGB224，来源与选择见data/manifest.json。须遵守原数据集使用条款。默认1个epoch共20步，是交付入口快速验证，不是完整收敛测试。历史长跑数据为3200/800图片，若复现历史长跑，须另备对应 data.npz 与 manifest.json，设置 DATA_DIR；不可把本目录的小子集结果当作历史长跑结果。

CINN 测试权重已托管到下述星河模型库。外部 CHECKPOINT_DIR 下提供 initial.pdparams 和 inference.pdparams，本次两者均为相同原始随机权重，不使用各端分别训练后的权重作推理对齐。大小和SHA256见asset_manifest.json；入口自动校验打包数据及权重。外部DATA_DIR由使用者选择，训练记录其manifest哈希。

## 运行

历史完整子集的文件列表、预处理、标签映射及 data.npz 大小/SHA256 见 外部资源包 full_dataset_reference.json；该较大的数据文件不包含在本次快速测试包内。

```bash
export CINN_RESOURCE_DIR=/path/to/mobilenetv5-resources
export CHECKPOINT_DIR=/path/to/local-weights
export PYTHON_BIN=/path/to/python
bash tests/cinn/mobilenetv5/run_model.sh
bash tests/cinn/mobilenetv5/run_model_cinn.sh
# 如需长跑，先准备原始3200/800数据集：
MODE=dynamic DATA_DIR=/path/to/full-subset EPOCHS=30 RESULTS_DIR=results/full bash tests/cinn/mobilenetv5/train.sh
MODE=cinn DATA_DIR=/path/to/full-subset EPOCHS=30 RESULTS_DIR=results/full bash tests/cinn/mobilenetv5/train.sh
MODE=dynamic RESULTS_DIR=results/infer1 bash tests/cinn/mobilenetv5/infer.sh
MODE=cinn RESULTS_DIR=results/infer1 bash tests/cinn/mobilenetv5/infer.sh
```

GPU_ID默认0。重复运行必须指定新的RESULTS_DIR，拒绝覆盖旧目录。train.sh支持RESUME指定可信训练状态（不要跨两端混用）。run_model.py负责调用train_impl.py或infer_impl.py；所有入口在本目录中自包含，不依赖个人绝对路径。

## 测量与验证

训练记录每步loss、train_ms及每epoch验证，输出config/environment、final权重和训练状态。独立推理以固定验证batch前向，预热5次后计时50次，GPU同步，排除预处理、H2D和首次编译；保存logits.npy与inference.json。CINN日志必须存在编译标志。

吞吐提升=(动态图ms/CINNms-1)×100%；FPS=batch×1000/ms。该短测只验证入口及固定权重数值，不能证明完整收敛或将一次短测当作稳定性能结论。运行产物和权重均不提交Git。

2026-10-07 交付入口回归通过：两端各1 epoch/20步，loss均有限，最大绝对差1.19e-6；共同权重独立推理logits最大绝对差1.67e-6。单次短测动态/CINN分别76.094/56.752ms，仅用于入口验证，不替代历史长程材料。独立推理使用随机初始权重，分类准确率不作为应用质量指标。

## Official repository entry

2026-11-06 local entry regression on develop c37ec6b with the two missing import-list parentheses fixed: both modes completed one epoch (20 steps) and independent inference. Loss max difference 1.19e-6; same-checkpoint logits max difference 1.67e-6. This is a short entry regression, not a new convergence run.

Training and inference implementation is shared in tests/cinn/_shared/; keep that directory when copying this entry.

Import PaddleClas from this checkout; no historical import-syntax patch is applied. External resources are not included: CINN_RESOURCE_DIR must contain the data/ and assets/ files in asset_manifest.json. CHECKPOINT_DIR supplies the matching weights. Download checkpoints using the links below; other assets still need to be obtained from the contributor. Follow the source dataset license. Historical measurements above are not new convergence results for this checkout.

## CINN checkpoint downloads

Model library: [PaddleClas/MobileNet-V5](https://aistudio.baidu.com/modelsdetail/45706/space).

- [initial.pdparams](https://git.aistudio.baidu.com/api/v1/repos/PaddleClas/MobileNet-V5/media/cinn%2Fmobilenetv5%2Fcheckpoint%2Finitial.pdparams): common training initialization for dynamic and CINN runs.
- [inference.pdparams](https://git.aistudio.baidu.com/api/v1/repos/PaddleClas/MobileNet-V5/media/cinn%2Fmobilenetv5%2Fcheckpoint%2Finference.pdparams): the same checkpoint must be used for both independent inference runs.
- [checkpoint_manifest.json](https://git.aistudio.baidu.com/api/v1/repos/PaddleClas/MobileNet-V5/media/cinn%2Fmobilenetv5%2Fcheckpoint_manifest.json): file sizes, SHA256 and test scope.

Save the two weight files under `CHECKPOINT_DIR`. The existing `asset_manifest.json` pins their hashes; do not substitute other weights from the model library. These are framework-test checkpoints, not a new task-quality evaluation. Configuration/tokenizer/data assets are separate and are not supplied by these weight downloads.
