OpenCV_DIR=path/to/opencv
PADDLE_LIB_DIR=path/to/paddle
CUDA_LIB_DIR=path/to/cuda
CUDNN_LIB_DIR=path/to/cudnn
TENSORRT_LIB_DIR=path/to/tensorrt
CONFIG_LIB_PATH=path/to/config/library
CLS_LIB_PATH=path/to/cls/library
CMP_STATIC=ON

BUILD_DIR=build
rm -rf ${BUILD_DIR}
mkdir ${BUILD_DIR}
cd ${BUILD_DIR}
cmake .. \
    -DWITH_MKL=ON \
    -DWITH_GPU=OFF \
    -DWITH_TENSORRT=OFF \
    -DOpenCV_DIR=${OpenCV_DIR} \
    -DPADDLE_LIB=${PADDLE_LIB_DIR} \
    -DCUDA_LIB=${CUDA_LIB_DIR} \
    -DCUDNN_LIB=${CUDNN_LIB_DIR} \
    -DCONFIG_LIB=${CONFIG_LIB_PATH} \
    -DCLS_LIB=${CLS_LIB_PATH} \
    -DCMP_STATIC=OFF

make -j
