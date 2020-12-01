
export CUDA_VISIBLE_DEVICES=6 

batch_size=128
python3 static/train.py \
        -c ../configs/ResNet/ResNet50_fp16.yml \
        -o TRAIN.batch_size=${batch_size} \
        -o validate=False \
        -o epochs=1 \
        -o TRAIN.data_dir=/ssd3/datasets/ILSVRC2012 \
        -o TRAIN.file_list=/ssd3/datasets/ILSVRC2012/train_list.txt\
        -o TRAIN.num_workers=4
        -o ARCHITECTURE.params.data_format=NHWC