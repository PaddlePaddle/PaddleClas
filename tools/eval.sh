python -m paddle.distributed.launch \
    --gpus="0" \
    tools/eval.py \
        -c ./configs/eval.yaml
