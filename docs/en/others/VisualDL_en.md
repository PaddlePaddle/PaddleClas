# Use VisualDL to visualize the training

---

## Catalogue

* [1. Preface](#1)
* [2. Use VisualDL in PaddleClas](#2)
    * [2.1 Set config and start training](#2.1)
    * [2.2 Start VisualDL](#2.2)

<a name='1'></a>
## 1. Preface
VisualDL, a visualization analysis tool of PaddlePaddle, provides a variety of charts to show the trends of parameters, and visualizes model structures, data samples, histograms of tensors, PR curves , ROC curves and high-dimensional data distributions. It enables users to understand the training process and the model structure more clearly and intuitively so as to optimize models efficiently. For more information, please refer to [VisualDL](https://github.com/PaddlePaddle/VisualDL/).

<a name='2'></a>
## 2. Use VisualDL in PaddleClas
Now PaddleClas support use VisualDL to visualize the changes of learning rate, loss, accuracy in training.

<a name='2.1'></a>
### 2.1 Set config and start training
You only need to set the field `Global.use_visualdl` to `True` in train config:

```yaml
# config.yaml
Global:
...
  use_visualdl: True
...
```

PaddleClas will save the VisualDL logs to subdirectory `vdl/` under the output directory specified by `Global.output_dir`. And then you just need to start training normally:

```shell
python3 tools/train.py -c config.yaml
```

<a name='2.2'></a>
### 2.2 Start VisualDL
After starting the training program, you can start the VisualDL service in a new terminal session:

```shell
 visualdl --logdir ./output/vdl/
```

In the above command, `--logdir` specify the directory of the VisualDL logs produced in training. VisualDL will traverse and iterate to find the subdirectories of the specified directory to visualize all the experimental results. You can also use the following parameters to set the IP and port number of the VisualDL service:

* `--host`：ip, default is 127.0.0.1
* `--port`：port, default is 8040

More information about the command，please refer to [VisualDL](https://github.com/PaddlePaddle/VisualDL/blob/develop/README.md#2-launch-panel).

Then you can enter the address `127.0.0.1:8840` and view the training process in the browser:


![](../../images/VisualDL/train_loss.png)

