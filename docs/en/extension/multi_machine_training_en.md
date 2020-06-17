# Distributed Training

Distributed deep neural networks training is highly efficient in PaddlePaddle. 
And it is one of the PaddlePaddle's core advantage technologies.
On image classification tasks, distributed training can achieve almost linear acceleration ratio.
[Fleet](https://github.com/PaddlePaddle/Fleet) is High-Level API for distributed training in PaddlePaddle. 
By using Fleet, a user can shift from local machine paddlepaddle code to distributed code easily.
In order to support both single-machine training and multi-machine training,
[PaddleClas](https://github.com/PaddlePaddle/PaddleClas) uses the Fleet API interface.
For more information about distributed training,
please refer to [Fleet API documentation](https://github.com/PaddlePaddle/Fleet/blob/develop/README.md).

