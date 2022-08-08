#!/bin/bash
BASEDIR=$(dirname "$0")

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

REPO_ROOT_PATH=$(readlinkf ${BASEDIR}/../)

config_files=$(find ${REPO_ROOT_PATH}/test_tipc/configs -name "*.txt")
for file in ${config_files}; do
   sed -i "s/Global.device:gpu/Global.device:xpu/g" $file
   sed -i "s/Global.use_gpu/Global.use_xpu/g" $file
done

config_files=$(find ${REPO_ROOT_PATH}/ppcls/configs -name "*.yaml")
for file in ${config_files}; do
   sed -i "s/device: gpu/device: xpu/g" $file
done

config_files=$(find ${REPO_ROOT_PATH}/deploy/configs -name "*.yaml")
for file in ${config_files}; do
   sed -i "s/use_gpu: True/use_xpu: True/g" $file
done
