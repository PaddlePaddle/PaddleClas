# Install PaddlePaddle

---

## Catalogue

- [1. Environment requirements](#1)
- [2.(Recommended) Prepare a docker environment](#2)
- [3. Install PaddlePaddle using pip](#3)
- [4. Verify installation](#4)

At present, **PaddleClas** requires **PaddlePaddle** version `>=2.0`. Docker is recomended to run Paddleclas, for more detailed information about docker and nvidia-docker, you can refer to the [tutorial](https://docs.docker.com/get-started/). If you do not want to use docker, you can skip section [2. (Recommended) Prepare a docker environment](#2), and go into section [3. Install PaddlePaddle using pip](#3).

<a name="1"></a>

## 1. Environment requirements

- python 3.x
- cuda >= 10.1 (necessary if paddlepaddle-gpu is used)
- cudnn >= 7.6.4 (necessary if paddlepaddle-gpu is used)
- nccl >= 2.1.2 (necessary distributed training/eval is used)
- gcc >= 8.2

**Recomends**:
* When CUDA version is 10.1, the driver version `>= 418.39`;
* When CUDA version is 10.2, the driver version `>= 440.33`;
* For more CUDA versions and specific driver versions, please refer to [link](https://docs.nvidia.com/deploy/cuda-compatibility/index.html).


<a name="2"></a>
## 2. (Recommended) Prepare a docker environment

* Switch to the working directory

```shell
cd /home/Projects
```

* Create docker container
The following commands will create a docker container named ppcls and map the current working directory to the `/paddle' directory in the container.

```shell
# For GPU users
sudo nvidia-docker run --name ppcls -v $PWD:/paddle --shm-size=8G --network=host -it paddlepaddle/paddle:2.1.0-gpu-cuda10.2-cudnn7 /bin/bash

# For CPU users
sudo docker run --name ppcls -v $PWD:/paddle --shm-size=8G --network=host -it paddlepaddle/paddle:2.1.0 /bin/bash
```

**Notices**:
* The first time you use this docker image, it will be downloaded automatically. Please be patient;
* The above command will create a docker container named ppcls, and there is no need to run the command again when using the container again;
* The parameter `--shm-size=8g` will set the shared memory of the container to 8g. If conditions permit, it is recommended to set this parameter to a larger value, such as `64g`;
* You can also access [DockerHub](https://hub.Docker.com/r/paddlepaddle/paddle/tags/) to obtain the image adapted to your machine;
* Exit / Enter the docker container:
    * After entering the docker container, you can exit the current container by pressing `Ctrl + P + Q` without closing the container;
    * To re-enter the container, use the following command:
    ```shell
    sudo Docker exec -it ppcls /bin/bash
    ```

<a name="3"></a>

## 3. Install PaddlePaddle using pip

If you want to use PaddlePaddle on GPU, you can use the following command to install PaddlePaddle.

```bash
pip install paddlepaddle-gpu --upgrade -i https://mirror.baidu.com/pypi/simple
```

If you want to use PaddlePaddle on CPU, you can use the following command to install PaddlePaddle.

```bash
pip install paddlepaddle --upgrade -i https://mirror.baidu.com/pypi/simple
```

**Note:**
* If you have already installed CPU version of PaddlePaddle and want to use GPU version now, you should uninstall CPU version of PaddlePaddle and then install GPU version to avoid package confusion.
* You can also compile PaddlePaddle from source code, please refer to [PaddlePaddle Installation tutorial](http://www.paddlepaddle.org.cn/install/quick) to more compilation options.

<a name="4"></a>

## 4. Verify Installation

```python
import paddle
paddle.utils.run_check()
```

Check PaddlePaddle version：

```bash
python -c "import paddle; print(paddle.__version__)"
```

Note:
* Make sure the compiled source code is later than PaddlePaddle2.0.
* Indicate `WITH_DISTRIBUTE=ON` when compiling, Please refer to [Instruction](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#id3) for more details.
* When running in docker, in order to ensure that the container has enough shared memory for dataloader acceleration of Paddle, please set the parameter `--shm-size=8g` at creating a docker container, if conditions permit, you can set it to a larger value.
