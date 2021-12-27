# Metric Learning
----
## 目录

* [1. 简介](#1)
* [2. 应用](#2)
* [3. 算法](#3)
    * [3.1 Classification based](#3.1)
    * [3.2 Pairwise based](#3.2)

<a name='1'></a>
## 1. 简介
   在机器学习中，我们经常会遇到度量数据间距离的问题。一般来说，对于可度量的数据，我们可以直接通过欧式距离(Euclidean Distance)，向量内积(Inner Product)或者是余弦相似度(Cosine Similarity)来进行计算。但对于非结构化数据来说，我们却很难进行这样的操作，如计算一段视频和一首音乐的匹配程度。由于数据格式的不同，我们难以直接进行上述的向量运算，但先验知识告诉我们 ED(laugh_video, laugh_music) < ED(laugh_video, blue_music), 如何去有效得表征这种”距离”关系呢? 这就是 Metric Learning 所要研究的课题。

   Metric learning 全称是 Distance Metric Learning，它是通过机器学习的形式，根据训练数据，自动构造出一种基于特定任务的度量函数。Metric Learning 的目标是学习一个变换函数（线性非线性均可）L，将数据点从原始的向量空间映射到一个新的向量空间，在新的向量空间里相似点的距离更近，非相似点的距离更远，使得度量更符合任务的要求，如下图所示。 Deep Metric Learning，就是用深度神经网络来拟合这个变换函数。
![example](../../images/ml_illustration.jpg)

<a name='2'></a>
## 2. 应用
   Metric Learning 技术在生活实际中应用广泛，如我们耳熟能详的人脸识别(Face Recognition)、行人重识别(Person ReID)、图像检索(Image Retrieval)、细粒度分类(Fine-grained classification)等。随着深度学习在工业实践中越来越广泛的应用，目前大家研究的方向基本都偏向于 Deep Metric Learning(DML).

   一般来说, DML 包含三个部分: 特征提取网络来 map embedding, 一个采样策略来将一个 mini-batch 里的样本组合成很多个 sub-set, 最后 loss function 在每个 sub-set 上计算 loss. 如下图所示：
   ![image](../../images/ml_pipeline.jpg)

<a name='3'></a>
## 3. 算法
   Metric Learning 主要有如下两种学习范式：
<a name='3.1'></a>
### 3.1 Classification based:  
   这是一类基于分类标签的 Metric Learning 方法。这类方法通过将每个样本分类到正确的类别中，来学习有效的特征表示，学习过程中需要每个样本的显式标签参与 Loss 计算。常见的算法有 [L2-Softmax](https://arxiv.org/abs/1703.09507), [Large-margin Softmax](https://arxiv.org/abs/1612.02295), [Angular Softmax](https://arxiv.org/pdf/1704.08063.pdf), [NormFace](https://arxiv.org/abs/1704.06369), [AM-Softmax](https://arxiv.org/abs/1801.05599), [CosFace](https://arxiv.org/abs/1801.09414), [ArcFace](https://arxiv.org/abs/1801.07698)等。
   这类方法也被称作是 proxy-based, 因为其本质上优化的是样本和一堆 proxies 之间的相似度。
<a name='3.2'></a>
### 3.2 Pairwise based:
   这是一类基于样本对的学习范式。他以样本对作为输入，通过直接学习样本对之间的相似度来得到有效的特征表示，常见的算法包括：[Contrastive loss](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf), [Triplet loss](https://arxiv.org/abs/1503.03832), [Lifted-Structure loss](https://arxiv.org/abs/1511.06452), [N-pair loss](https://papers.nips.cc/paper/2016/file/6b180037abbebea991d8b1232f8a8ca9-Paper.pdf), [Multi-Similarity loss](https://arxiv.org/pdf/1904.06627.pdf)等

2020 年发表的[CircleLoss](https://arxiv.org/abs/2002.10857)，从一个全新的视角统一了两种学习范式，让研究人员和从业者对 Metric Learning 问题有了更进一步的思考。
