python -m paddle.distributed.launch \
    --selected_gpus="0" \
    tools/eval.py \
        -c ./configs/eval.yaml
