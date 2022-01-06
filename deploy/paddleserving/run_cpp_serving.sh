#run cls server:
nohup python3 -m paddle_serving_server.serve --model ResNet50_vd_serving --port 9292 &
