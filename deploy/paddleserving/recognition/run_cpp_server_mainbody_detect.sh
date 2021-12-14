nohup python3 -m paddle_serving_server.serve \
--model picodet_PPLCNet_x2_5_mainbody_lite_v1.0_serving \
 --port 9293 >>log_mainbody_detection.txt 1&>2 &