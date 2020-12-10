python3.7 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/eval.py \
        -c ./configs/ResNet/ResNet50.yaml \
        -o pretrained_model="./ResNet50_pretrained" \
        -o use_gpu=True
