model_item=ResNet50
bs_item=256
fp_item=pure_fp16
run_mode=DP
device_num=N4C32
max_epochs=32
num_workers=8

# get data
bash test_tipc/static/${model_item}/benchmark_common/prepare.sh

cd ./dataset/ILSVRC2012
cat train_list.txt >> tmp
for i in {1..10}; do cat tmp >> train_list.txt; done
cd ../../

# run
bash test_tipc/static/${model_item}/benchmark_common/run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_mode} ${device_num} ${max_epochs} ${num_workers} 2>&1;
