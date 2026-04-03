import time
from typing import List, Tuple, Union

import paddle
from tqdm import tqdm


def benchmark(
    model: paddle.nn.Module,
    device: str = "gpu:0",
    input_size: Tuple[int] = (3, 224, 224),
    batch_size: int = 64,
    runs: int = 40,
    throw_out: float = 0.25,
    use_fp16: bool = False,
    verbose: bool = False,
) -> float:
    """
    Benchmark the given model with random inputs at the given batch size.

    Args:
     - model: the module to benchmark
     - device: the device to use for benchmarking
     - input_size: the input size to pass to the model e.g., (ch, h, w) or (ch, t, h, w)
     - batch_size: the batch size to use for evaluation
     - runs: the number of total runs to do
     - throw_out: the percentage of runs to throw out at the start of testing
     - use_fp16: whether or not to benchmark with float16 and autocast
     - verbose: whether or not to use tqdm to print progress / print throughput at end

    Returns:
     - the throughput measured in images / second
    """
    is_cuda = "gpu" in device.lower()
    model = model.eval()
    paddle.device.set_device(device)
    input_tensor = paddle.rand([batch_size, *input_size])
    if use_fp16:
        input_tensor = input_tensor.cast(paddle.float16)
    warm_up = int(runs * throw_out)
    total = 0
    start = time.time()
    with paddle.amp.auto_cast(enable=use_fp16):
        with paddle.no_grad():
            for i in tqdm(range(runs), disable=not verbose, desc="Benchmarking"):
                if i == warm_up:
                    if is_cuda:
                        paddle.device.cuda.synchronize()
                    total = 0
                    start = time.time()
                model(input_tensor)
                total += batch_size
    if is_cuda:
        paddle.device.cuda.synchronize()
    end = time.time()
    elapsed = end - start
    throughput = total / elapsed
    if verbose:
        print(f"Throughput: {throughput:.2f} im/s")
    return throughput
