
# 标题

|任务名称 | SSLD蒸馏Teacher离线化 | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | lxh、xzr | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2023-9-21 | 
|版本号 | V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | 2.5.0 | 
|文件名 | 20230921_offline-distillation_rfc.md | 

# 一、概述
## 1、相关背景
近年来，深度神经网络在计算机视觉、自然语言处理等领域被验证是一种极其有效的解决问题的方法。通过构建合适的神经网络，加以训练，最终网络模型的性能指标基本上都会超过传统算法。在数据量足够大的情况下，通过合理构建网络模型的方式增加其参数量，可以显著改善模型性能，但是这又带来了模型复杂度急剧提升的问题。大模型在实际场景中使用的成本较高。深度神经网络一般有较多的参数冗余，目前有几种主要的方法对模型进行压缩，减小其参数量。如裁剪、量化、知识蒸馏等，其中知识蒸馏是指使用教师模型(teacher model)去指导学生模型(student model)学习特定任务，保证小模型在参数量不变的情况下，得到比较大的性能提升，甚至获得与大模型相似的精度指标。
## 2、功能目标
PaddleClas提供了一种简单高效的SSLD蒸馏方案，可以大幅度提升模型的精度。但是蒸馏过程中，每张图片均需要经过教师模型，导致整体训练时间比较久。本赛题主要内容是将PaddleClas中的SSLD蒸馏Teacher离线化，即教师模型将数据的预测标签进行存储，学生模型用来学习离线的教师模型的输出。最终SSLD蒸馏Teacher离线化的精度和标准SSLD精度打平，训练速度快1倍。
## 3、意义
教师模型将数据的预测标签进行存储，学生模型用来学习离线的教师模型的输出。提高SSLD模型蒸馏速度。

# 二、飞桨现状
对飞桨框架目前支持此功能的现状调研，如果不支持此功能，是否有其他可绕过的方式.


# 三、业内方案调研
描述业内深度学习框架如何实现此功能，包括与此功能相关的现状、未来趋势；调研的范围包括不限于TensorFlow、PyTorch、NumPy等。

# 四、对比分析
对第三部分调研的方案进行对比**评价**和**对比分析**，论述各种方案的优劣势。

# 五、设计思路与实现方案

## 1、主体设计思路与折衷
核心思想是通过结合老师模型与学生模型在线训练的方法变为离线训练，所谓离线的意思是替代数据or筛选出来的无标签数据预先输入老师模型，得到对应的label，构成新的替代训练数据，存储本地，再用于学生模型的训练。

# 六、测试和验收的考量
预期提到一倍以上的训练速度，同时考虑在JS损失的基础上，添加新的损失，提高模型蒸馏的acc。

# 七、影响面
需要进一步讨论的问题，开放性问题，有争议问题；对其他模块是否有影响。
## 对用户的影响
用户能明显察觉蒸馏速度的提升。
## 对二次开发用户的影响
无
## 对框架架构的影响
## 对性能的影响
## 对比业内深度学习框架的差距与优势的影响
## 其他风险

# 八、排期规划
十月二十号预期完成

# 名词解释

# 附件及参考资料
训练代码：https://github.com/PaddlePaddle/PaddleClas
参考文献： @article{Cui_Guo_Du_He_Li_Wu_Liu_Wen_Huang_Hu_et al._2021,  
 title={Beyond Self-Supervision: A Simple Yet Effective Network Distillation Alternative to Improve Backbones.}, 
 journal={Cornell University - arXiv,Cornell University - arXiv}, 
 author={Cui, Cheng and Guo, Ruoyu and Du, Yuning and He, Dongliang and Li, Fu and Wu, Zewu and Liu, Qiwen and Wen, Shilei and Huang, Jizhou and Hu, Xiaoguang and Yu, Dianhai and Ding, Errui and Ma, Yanjun}, 
 year={2021}, 
 month={Mar}, 
 language={en-US} 
 }
