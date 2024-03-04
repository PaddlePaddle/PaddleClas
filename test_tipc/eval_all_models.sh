#!/bin/bash
source ./test_tipc/common_func.sh

# function: Extract dataset name from eval.log
extract_dataset_name() {
    local file_path="$1"
    if [ ! -f "$file_path" ]; then
        return 1
    fi
    local result=$(grep "class_id_map_file" "$file_path" | \
                 awk -F 'class_id_map_file' '{print $2}' | \
                 grep -o 'ppcls/utils/.*_label_list.txt' | \
                 sed 's/^ppcls\/utils\///;s/_label_list.txt$//')
    if [ "$result" = "imagenet1k" ]; then
        echo "ImageNet1K"
    else
        echo "$result"
    fi
}

# function: Extract top1 accuracy from the last line of eval.log
extract_top1() {
    local file_path="$1"
    if [ ! -f "$file_path" ]; then
        return 1
    fi
    local last_line=$(tail -n 1 "$file_path")
    local top1_acc=$(echo "$last_line" | \
                    awk -F 'top1:' '{print $2}' | \
                    awk -F ',' '{print $1}' | \
                    sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    echo "$top1_acc"
}

log_dir="./eval_logs"
md_path="./eval_list.md"
mkdir -p "$log_dir"
touch "$md_path"

if [ ! -s "$md_path" ]; then
    echo "| Model | Dataset | Top1 Acc |" >> "$md_path"
    echo "| :--: | :--: | :--: |" >> "$md_path"
fi

target_dir="./test_tipc/configs"
skip_num=0
eval_num=0
# 1. Traverse the config directory of each model
for target_dir_path in "$target_dir"/*; do
    # 2. Get the path of train_infer_python.txt
    for txt_path in "$target_dir_path"/*train_infer_python.txt; do
        if [ -f "$txt_path" ]; then       
            # 3. Use python3.9
            sed -i "s/python3.7/python3.9/g" "$txt_path"
            sed -i "s/python3.10/python3.9/g" "$txt_path" 

            # 4. Parser params
            dataline=$(cat $txt_path)  
            IFS=$'\n'
            lines=(${dataline})
            model_name=$(func_parser_value "${lines[1]}")
            python=$(func_parser_value "${lines[2]}")
            use_gpu_key=$(func_parser_key "${lines[4]}")
            use_gpu_value=$(func_parser_value "${lines[4]}")
            eval_py=$(func_parser_value "${lines[23]}")
            eval_key=$(func_parser_key "${lines[24]}")
            eval_value=$(func_parser_value "${lines[24]}")
            set_eval_params=$(func_set_params "${eval_key}" "${eval_value}")
            set_use_gpu=$(func_set_params "${use_gpu_key}" "${use_gpu_value}")
            log_path="${log_dir}/${model_name}_eval.log"

            # 5. Skip models that already listed in the md file or don't have eval.py
            if grep -q " $model_name " "$md_path" || [ "${eval_py}" == "null" ]; then
                ((skip_num++))
                continue
            fi

            # 6. Prepare data -> eval -> remove weight file
            # With "-o Arch.pretrained=True", the weights will be downloaded
            # to "/root/.paddleclas/weights" and loaded automatically.
            prepare_cmd="bash test_tipc/prepare.sh "$txt_path" 'lite_train_lite_infer'"
            eval_cmd="${python} ${eval_py} ${set_use_gpu} ${set_eval_params} -o Arch.pretrained=True > ${log_path}"
            rm_weight_cmd="rm -rf /root/.paddleclas/weights/*"
            eval ${prepare_cmd}
            eval ${eval_cmd}
            status_check $? "${eval_cmd}" "./eval_status.log" "${model_name}"
            eval ${rm_weight_cmd}

            # 7. Write evaluation results to md file
            dataset_name=$(extract_dataset_name "${log_path}")
            top1_acc=$(extract_top1 "${log_path}")
            echo "| $model_name | $dataset_name | $top1_acc |" >> "$md_path"            
            ((eval_num++))
            sleep 5
        fi
    done
done

model_num=$(expr $skip_num + $eval_num)
echo "Done."
echo "Skip ${skip_num} models."
echo "Eval ${eval_num} models."
echo "${model_num} models in total."