from tools.infer import utils
import numpy as np
import time

from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor


class System(object):
    def __init__(self, args):
        # HALF precission predict only work when using tensorrt
        if args.use_fp16 is True:
            assert args.use_tensorrt is True

        self.args = args
        self.operators = self._create_operators()
        self.predictor = self._create_predictor()

    def _create_predictor(self):
        config = AnalysisConfig(self.args.model_file, self.args.params_file)

        if self.args.use_gpu:
            config.enable_use_gpu(self.args.gpu_mem, 0)
        else:
            config.disable_gpu()

        config.disable_glog_info()
        config.switch_ir_optim(self.args.ir_optim)  # default true
        if self.args.use_tensorrt:
            config.enable_tensorrt_engine(
                precision_mode=AnalysisConfig.Precision.Half
                if self.args.use_fp16 else AnalysisConfig.Precision.Float32,
                max_batch_size=self.args.batch_size)

        config.enable_memory_optim()
        # use zero copy
        config.switch_use_feed_fetch_ops(False)
        predictor = create_paddle_predictor(config)

        return predictor

    def _create_operators(self):
        size = 224
        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]
        img_scale = 1.0 / 255.0

        resize_op = utils.ResizeImage(resize_short=256)
        crop_op = utils.CropImage(size=(size, size))
        normalize_op = utils.NormalizeImage(
            scale=img_scale, mean=img_mean, std=img_std)
        totensor_op = utils.ToTensor()
        return [resize_op, crop_op, normalize_op, totensor_op]

    def _preprocess(self, data, ops):
        for op in ops:
            data = op(data)

        return data

    def __call__(self, img, top_k):
        input_names = self.predictor.get_input_names()
        input_tensor = self.predictor.get_input_tensor(input_names[0])

        output_names = self.predictor.get_output_names()
        output_tensor = self.predictor.get_output_tensor(output_names[0])

        inputs = self._preprocess(img, self.operators)
        inputs = utils.preprocess(img, self.args)
        inputs = np.expand_dims(
            inputs, axis=0).repeat(
                self.args.batch_size, axis=0).copy()
        input_tensor.copy_from_cpu(inputs)

        self.predictor.zero_copy_run()

        output = output_tensor.copy_to_cpu()
        output = output.flatten()
        cls = np.argmax(output)
        scores = np.sort(output)[-1:-top_k - 1:-1]
        return (cls, scores)
