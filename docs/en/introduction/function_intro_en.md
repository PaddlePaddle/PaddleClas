## Features of PaddleClas

PaddleClas is an image recognition toolset for industry and academia, helping users train better computer vision models and apply them in real scenarios. Specifically, it contains the following core features.

- Practical image recognition system: Integrate detection, feature learning, and retrieval modules to be applicable to all types of image recognition tasks. Four sample solutions are provided, including product recognition, vehicle recognition, logo recognition, and animation character recognition.
- Rich library of pre-trained models: Provide a total of 175 ImageNet pre-trained models of 36 series, among which 7 selected series of models support fast structural modification.
- Comprehensive and easy-to-use feature learning components: 12 metric learning methods are integrated and can be combined and switched at will through configuration files.
- SSLD knowledge distillation: The 14 classification pre-training models generally improved their accuracy by more than 3%; among them, the ResNet50_vd model achieved a Top-1 accuracy of 84.0% on the Image-Net-1k dataset and the Res2Net200_vd pre-training model achieved a Top-1 accuracy of 85.1%.
- Data augmentation: Provide 8 data augmentation algorithms such as AutoAugment, Cutout, Cutmix, etc. with the detailed introduction, code replication, and evaluation of effectiveness in a unified experimental environment.

![](../../images/recognition.gif)

For more information about the quick start of image recognition, algorithm details, model training and evaluation, and prediction and deployment methods, please refer to the [README Tutorial](../../../README_ch.md) on home page.
