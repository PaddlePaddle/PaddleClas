OPENCV_DIR=/work/project/project/test/opencv-3.4.7/opencv3
LIB_DIR=/work/project/project/test/paddle_inference/
CUDA_LIB_DIR=/usr/local/cuda/lib64
CUDNN_LIB_DIR=/usr/lib/x86_64-linux-gnu/
FAISS_DIR=/work/project/project/test/faiss/faiss_install
FAISS_WITH_MKL=OFF

BUILD_DIR=build
rm -rf ${BUILD_DIR}
mkdir ${BUILD_DIR}
cd ${BUILD_DIR}
cmake .. \
    -DPADDLE_LIB=${LIB_DIR} \
    -DWITH_MKL=ON \
    -DWITH_GPU=OFF \
    -DWITH_STATIC_LIB=OFF \
    -DUSE_TENSORRT=OFF \
    -DOPENCV_DIR=${OPENCV_DIR} \
    -DCUDNN_LIB=${CUDNN_LIB_DIR} \
    -DCUDA_LIB=${CUDA_LIB_DIR} \
    -DFAISS_DIR=${FAISS_DIR} \
    -DFAISS_WITH_MKL=${FAISS_WITH_MKL}

make -j
cd ..
