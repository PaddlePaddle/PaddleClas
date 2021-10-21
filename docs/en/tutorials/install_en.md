# Installation

---
This tutorial introduces how to install PaddleClas and its requirements.

## 1. Install PaddlePaddle

`PaddlePaddle 2.1` or later is required for PaddleClas. You can use the following steps to install PaddlePaddle.

### 1.1 Environment requirements

- python 3.x
- cuda >= 10.1 (necessary if you want to use paddlepaddle-gpu)
- cudnn >= 7.6.4 (necessary if you want to use paddlepaddle-gpu)
- nccl >= 2.1.2 (necessary if you want the use distributed training/eval)
- gcc >= 8.2

Docker is recomended to run Paddleclas, for more detailed information about docker and nvidia-docker, you can refer to the [tutorial](https://www.runoob.com/docker/docker-tutorial.html).

When you use cuda10.1, the driver version needs to be larger or equal than 418.39. When you use cuda10.2, the driver version needs to be larger or equal than 440.33. For more cuda versions and specific driver versions, you can refer to the [link](https://docs.nvidia.com/deploy/cuda-compatibility/index.html).

If you do not want to use docker, you can skip section 1.2 and go into section 1.3 directly.


### 1.2 (Recommended) Prepare a docker environment. The first time you use this docker image, it will be downloaded automatically. Please be patient.


```
# Switch to the working directory
cd /home/Projects
# You need to create a docker container for the first run, and do not need to run the current command when you run it again
# Create a docker container named ppcls and map the current directory to the /paddle directory of the container
# It is recommended to set a shared memory greater than or equal to 8G through the --shm-size parameter
sudo docker run --name ppcls -v $PWD:/paddle --shm-size=8G --network=host -it paddlepaddle/paddle:2.1.0 /bin/bash

# Use the following command to create a container if you want to use GPU in the container
sudo nvidia-docker run --name ppcls -v $PWD:/paddle --shm-size=8G --network=host -it paddlepaddle/paddle:2.1.0-gpu-cuda10.2-cudnn7 /bin/bash
```

You can also visit [DockerHub](https://hub.docker.com/r/paddlepaddle/paddle/tags/) to get more docker images.

```
# use ctrl+P+Q to exit docker, to re-enter docker using the following command:
sudo docker exec -it ppcls /bin/bash
```

### 1.3 Install PaddlePaddle using pip

If you want to use PaddlePaddle on GPU, you can use the following command to install PaddlePaddle.

```bash
pip3 install paddlepaddle-gpu --upgrade -i https://mirror.baidu.com/pypi/simple
```

If you want to use PaddlePaddle on CPU, you can use the following command to install PaddlePaddle.

```bash
pip3 install paddlepaddle --upgrade -i https://mirror.baidu.com/pypi/simple
```

**Note:**
* If you have already installed CPU version of PaddlePaddle and want to use GPU version now, you should uninstall CPU version of PaddlePaddle and then install GPU version to avoid package confusion.
* You can also compile PaddlePaddle from source code, please refer to [PaddlePaddle Installation tutorial](http://www.paddlepaddle.org.cn/install/quick) to more compilation options.

### 1.4 Verify Installation process

```python
import paddle
paddle.utils.run_check()
```

Check PaddlePaddle version：

```bash
python3 -c "import paddle; print(paddle.__version__)"
```

Note:
- Make sure the compiled source code is later than PaddlePaddle2.0.
- Indicate **WITH_DISTRIBUTE=ON** when compiling, Please refer to [Instruction](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#id3) for more details.
- When running in docker, in order to ensure that the container has enough shared memory for dataloader acceleration of Paddle, please set the parameter `--shm_size=8g` at creating a docker container, if conditions permit, you can set it to a larger value.



## 2. Install PaddleClas

### 2.1 Clone PaddleClas source code

```
git clone https://github.com/PaddlePaddle/PaddleClas.git -b develop
```

If it is too slow for you to download from github, you can download PaddleClas from gitee. The command is as follows.

```bash
git clone https://gitee.com/paddlepaddle/PaddleClas.git -b develop
```

### 2.2 Install requirements

PaddleClas dependencies are listed in file `requirements.txt`, you can use the following command to install the dependencies.

```
pip3 install --upgrade -r requirements.txt -i https://mirror.baidu.com/pypi/simple
```
