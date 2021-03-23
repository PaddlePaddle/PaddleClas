# 使用VisualDL可视化训练过程

## 前言
VisualDL是飞桨可视化分析工具，以丰富的图表呈现训练参数变化趋势、模型结构、数据样本、高维数据分布等。可帮助用户更清晰直观地理解深度学习模型训练过程及模型结构，进而实现高效的模型优化。更多细节请查看[VisualDL](https://github.com/PaddlePaddle/VisualDL/)。

## 在PaddleClas中使用VisualDL
现在PaddleClas支持在训练阶段使用VisualDL查看训练过程中学习率（learning rate）、损失值（loss）以及准确率（accuracy）的变化情况。

### 设置config文件并启动训练
在PaddleClas中使用VisualDL，只需在训练配置文件（config文件）添加如下字段：

```yaml
# confit.txt
vdl_dir: "./vdl.log"
```
`vdl_dir` 用于指定VisualDL用于保存log信息的目录。

然后正常启动训练即可：

```shell
python3 tools/train.py -c config.txt
```

### 启动VisualDL
在启动训练程序后，可以在新的终端session中启动VisualDL服务：

```shell
 visualdl --logdir ./vdl.log
 ```

上述命令中，参数`--logdir`用于指定日志目录，VisualDL将遍历并且迭代寻找指定目录的子目录，将所有实验结果进行可视化。也同样可以使用下述参数设定VisualDL服务的ip及端口号：
* `--host`：设定IP，默认为127.0.0.1
* `--port`：设定端口，默认为8040

更多参数信息，请查看[VisualDL](https://github.com/PaddlePaddle/VisualDL/blob/develop/README_CN.md#2-%E5%90%AF%E5%8A%A8%E9%9D%A2%E6%9D%BF)。

在启动VisualDL后，即可在浏览器中查看训练过程，输入地址`127.0.0.1:8840`：

<div align="center">
    <img src="../../images/VisualDL/train_loss.png" width="400">
</div>
