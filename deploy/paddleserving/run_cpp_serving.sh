gpu_id=$1

# ResNet50_vd CPP serving script
if [[ -n "${gpu_id}" ]]; then
    nohup python3.7 -m paddle_serving_server.serve \
    --model ./ResNet50_vd_serving \
    --op GeneralClasOp \
    --port 9292 &
else
    nohup python3.7 -m paddle_serving_server.serve \
    --model ./ResNet50_vd_serving \
    --op GeneralClasOp \
    --port 9292 --gpu_id="${gpu_id}" &
fi
