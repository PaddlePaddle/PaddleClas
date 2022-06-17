gpu_id=$1

# PP-ShiTu CPP serving script
if [[ -n "${gpu_id}" ]]; then
    nohup python3.7 -m paddle_serving_server.serve \
    --model ../../models/picodet_PPLCNet_x2_5_mainbody_lite_v1.0_serving ../../models/general_PPLCNet_x2_5_lite_v1.0_serving \
    --op GeneralPicodetOp GeneralFeatureExtractOp \
    --port 9400 --gpu_id="${gpu_id}" > log_PPShiTu.txt 2>&1 &
else
    nohup python3.7 -m paddle_serving_server.serve \
    --model ../../models/picodet_PPLCNet_x2_5_mainbody_lite_v1.0_serving ../../models/general_PPLCNet_x2_5_lite_v1.0_serving \
    --op GeneralPicodetOp GeneralFeatureExtractOp \
    --port 9400 > log_PPShiTu.txt 2>&1 &
fi
