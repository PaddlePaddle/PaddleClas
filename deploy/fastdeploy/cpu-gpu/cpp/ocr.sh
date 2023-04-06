rm -rf build
mkdir build

cd build

#/xieyunyao/project/FastDeploy

cmake .. -DFASTDEPLOY_INSTALL_DIR=/xieyunyao/project/fastdeploy-linux-x64-gpu-0.0.0

make -j
