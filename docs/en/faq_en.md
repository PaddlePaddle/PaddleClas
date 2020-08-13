# FAQ

>>
* Why are the metrics different for different cards?
* A: Fleet is the default option for the use of PaddleClas. Each GPU card is taken as a single trainer and deals with different images, which cause the final small difference. Single card evalution is suggested to get the accurate results if you use `tools/eval.py`. You can also use  `tools/eval_multi_platform.py` to evalute the models on multiple GPU cards, which is also supported on Windows and CPU.


>>
* Q: Why `Mixup` or `Cutmix` is not used even if I have already add the data operation in the configuration file?
* A: When using `Mixup` or `Cutmix`, you also need to add `use_mix: True` in the configuration file to make it work properly.


>>
* Q: During evaluation and inference, pretrained model address is assgined, but the weights can not be imported. Why?
* A: Prefix of the pretrained model is needed. For example, if the pretained weights are located in `output/ResNet50_vd/19`, with the filename `output/ResNet50_vd/19/ppcls.pdparams`, then `pretrained_model` in the configuration file needs to be `output/ResNet50_vd/19/ppcls`.

>>
* Q: Why are the metrics 0.3% lower than that shown in the model zoo for `EfficientNet` series of models?
* A: Resize method is set as `Cubic` for `EfficientNet`(interpolation is set as 2 in OpenCV), while other models are set as `Bilinear`(interpolation is set as None in OpenCV). Therefore, you need to modify the interpolation explicitly in `ResizeImage`. Specifically, the following configuration is a demo for EfficientNet.

```
VALID:
    batch_size: 16
    num_workers: 4
    file_list: "./dataset/ILSVRC2012/val_list.txt"
    data_dir: "./dataset/ILSVRC2012/"
    shuffle_seed: 0
    transforms:
        - DecodeImage:
            to_rgb: True
            to_np: False
            channel_first: False
        - ResizeImage:
            resize_short: 256
            interpolation: 2
        - CropImage:
            size: 224
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''
        - ToCHWImage:
```

>>
* Q: What should I do if I want to transform the weights' format from `pdparams` to an earlier version(before Paddle1.7.0), which consists of the scattered files?
* A: You can use `fluid.load` to load the `pdparams` weights and use `fluid.io.save_vars` to save the weights as scattered files. The demo is as follows. Finally all the scattered files will be saved in the path `path_to_save_var`.
```
fluid.load(
        program=infer_prog, model_path=args.pretrained_model, executor=exe)
state = fluid.io.load_program_state(args.pretrained_model)
def exists(var):
    return var.name in state
fluid.io.save_vars(exe, "./path_to_save_var", infer_prog, predicate=exists)
```


>>
* Q: The error occured when using visualdl under python2, shows that: `TypeError: __init__() missing 1 required positional argument: 'sync_cycle'`.
* A: `Visualdl` is only supported on python3 as now, whose version needs also be higher than `2.0`. If your visualdl version is lower than 2.0, you can also install visualdl 2.0 by `pip3 install visualdl==2.0.0b8 -i https://mirror.baidu.com/pypi/simple`.
