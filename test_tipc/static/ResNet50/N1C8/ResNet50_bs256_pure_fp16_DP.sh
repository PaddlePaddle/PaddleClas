model_item=ResNet50
bs_item=256
fp_item=pure_fp16
run_mode=DP
device_num=N1C8
max_epochs=8
num_workers=8

# get data
bash test_tipc/static/${model_item}/benchmark_common/prepare.sh
# run
bash test_tipc/static/${model_item}/benchmark_common/run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_mode} ${device_num} ${max_epochs} ${num_workers} 2>&1;
