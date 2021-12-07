import onnx
import onnxruntime


def get_onnx_engine(onnx_model_path, num_thread=1):
    so = onnxruntime.SessionOptions()
    so.intra_op_num_threads = num_thread
    so.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    onnx_model = onnx.load_model(onnx_model_path)
    sess = onnxruntime.InferenceSession(onnx_model.SerializeToString(), sess_options=so)
    return sess


# class ONNXEngine(object):
#     def __init__(self, config_dict):



