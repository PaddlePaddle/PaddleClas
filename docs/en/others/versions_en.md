# Version Updates

------

## Catalogue

- [1. v2.3](#1)
- [2. v2.2](#2)

<a name='1'></a>

## 1. v2.3

- Model Update
  - Add pre-training weights for lightweight models, including detection models and feature models
  - Release PP-LCNet series of models, which are self-developed ones designed to run on CPU
  - Enable SwinTransformer, Twins, and Deit to support direct training from scrach to achieve thesis accuracy.
- Basic framework capabilities
  - Add DeepHash module, which supports feature model to directly export binary features
  - Add PKSampler, which tackles the problem that feature models cannot be trained by multiple machines and cards
  - Support PaddleSlim: support quantization, pruning training, and offline quantization of classification models and feature models
  - Enable legendary models to support intermediate model output
  - Support multi-label classification training
- Inference Deployment
  - Replace the original feature retrieval library with Faiss to improve platform adaptability
  - Support PaddleServing: support the deployment of classification models and image recognition process
- Versions of the Recommendation Library
  - python: 3.7
  - PaddlePaddle: 2.1.3
  - PaddleSlim: 2.2.0
  - PaddleServing: 0.6.1

<a name='2'></a>

## 2. v2.2

- Model Updates
  - Add models including LeViT, Twins, TNT, DLA, HardNet, RedNet, and SwinTransfomer
- Basic framework capabilities
  - Divide the classification models into two categories
    - legendary models: introduce TheseusLayer base class, add the interface to modify the network function, and support the networking data truncation and output
    - model zoo: other common classification models
  - Add the support of Metric Learning algorithm
    - Add a variety of related loss algorithms, and the basic network module gears (allow the combination with backbone and loss) for convenient use
    - Support both the general classification and metric learning-related training
  - Support static graph training
  - Classification training with dali acceleration supported
  - Support fp16 training
- Application Updates
  - Add specific application cases and related models of product recognition, vehicle recognition (vehicle fine-grained classification, vehicle ReID), logo recognition, animation character recognition
  - Add a complete pipeline for image recognition, including detection module, feature extraction module, and vector search module
- Inference Deployment
  - Add Mobius, Baidu's self-developed vector search module, to support the inference deployment of the image recognition system
  - Image recognition, build feature library that allows batch_size>1
- Documents Update
  - Add image recognition related documents
  - Fix bugs in previous documents
- Versions of the Recommendation Library
  - python: 3.7
  - PaddlePaddle: 2.1.2
