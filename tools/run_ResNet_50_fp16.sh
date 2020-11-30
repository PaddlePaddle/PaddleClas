
export FLAGS_conv_workspace_size_limit=4000 #MB
export FLAGS_cudnn_exhaustive_search=1
export FLAGS_cudnn_batchnorm_spatial_persistent=1

CUDA_VISIBLE_DEVICES=7 python3 static/train.py -c ../configs/ResNet/ResNet50.yaml -o TRAIN.batch_size=64

