# Use VisualDL to visualize the training

## Preface
VisualDL, a visualization analysis tool of PaddlePaddle, provides a variety of charts to show the trends of parameters, and visualizes model structures, data samples, histograms of tensors, PR curves , ROC curves and high-dimensional data distributions. It enables users to understand the training process and the model structure more clearly and intuitively so as to optimize models efficiently. For more information, please refer to [VisualDL](https://github.com/PaddlePaddle/VisualDL/).

## Use VisualDL in PaddleClas
Now PaddleClas support use VisualDL to visualize the changes of learning rate, loss, accuracy in training.

### Set config and start training
You only need to set the `vdl_dir` field in train config:

```yaml
# confit.txt
vdl_dir: "./vdl.log"
```

`vdl_dir`: Specify the directory where VisualDL stores logs.

Then normal start training:

```shell
python3 tools/train.py -c config.txt
```

### Start VisualDL
After starting the training program, you can start the VisualDL service in the new terminal session:

```shell
 visualdl --logdir ./vdl.log
```

In the above command, `--logdir` specify the logs directory. VisualDL will traverse and iterate to find the subdirectories of the specified directory to visualize all the experimental results. You can also use the following parameters to set the IP and port number of the VisualDL service:

* `--host`：ip, default is 127.0.0.1
* `--port`：port, default is 8040

More information about the command，please refer to [VisualDL](https://github.com/PaddlePaddle/VisualDL/blob/develop/README.md#2-launch-panel).

Then you can enter the address `127.0.0.1:8840` and view the training process in the browser:

<div align="center">
    <img src="../../images/VisualDL/train_loss.png" width="400">
</div>
