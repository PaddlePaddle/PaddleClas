#!/bin/bash
source test_tipc/common_func.sh

# always use the lite_train_lite_infer mode to speed. Modify the config file.
MODE=lite_train_lite_infer
BASEDIR=$(dirname "$0")

FILENAME=$1
sed -i 's/gpu_list.*$/gpu_list:0/g' $FILENAME
sed -i '23,$d' $FILENAME
sed -i 's/-o Global.device:.*$/-o Global.device:cpu/g' $FILENAME
sed -i '16s/$/ -o Global.print_batch_step=1/' ${FILENAME}


# get the log path.
IFS=$'\n'
dataline=$(cat ${FILENAME})
lines=(${dataline})
model_name=$(func_parser_value "${lines[1]}")
LOG_PATH="./test_tipc/output/${model_name}/${MODE}"
rm -rf $LOG_PATH
mkdir -p ${LOG_PATH}

# start dygraph train
dygraph_output=$LOG_PATH/dygraph_output.txt
sed -i '15ctrainer:norm_train' ${FILENAME}
cmd="bash test_tipc/test_train_inference_python.sh ${FILENAME} $MODE >$dygraph_output 2>&1"
echo $cmd
eval $cmd

# start dy2static train
dy2static_output=$LOG_PATH/dy2static_output.txt
sed -i '15ctrainer:to_static_train' ${FILENAME}
cmd="bash test_tipc/test_train_inference_python.sh ${FILENAME} $MODE >$dy2static_output 2>&1"
echo $cmd
eval $cmd

# analysis and compare the losses. 
dyout=`cat $dy2static_output | python3 test_tipc/extract_loss.py -v 'Iter:' -e 'loss: {%f},'`
stout=`cat $dygraph_output | python3 test_tipc/extract_loss.py -v 'Iter:' -e 'loss: {%f},'  `
echo $dyout
echo $stout
if [ "$dyout" = "" ]; then
    echo "Failed to run model."
    exit -1
fi
if [ "$dyout" = "$stout" ]; then 
    echo "Successful Run Dy2static."
    exit 0
else
    echo "Loss is not equal."
    echo "Dygraph Loss is: " 
    echo $dyout
    echo "Dy2Static Loss is: "
    echo $stout
    exit -1
fi
