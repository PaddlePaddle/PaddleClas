OpenCV_DIR=path/to/opencv
PADDLE_LIB_DIR=path/to/paddle

BUILD_DIR=./lib/build
rm -rf ${BUILD_DIR}
mkdir ${BUILD_DIR}
cd ${BUILD_DIR}
cmake .. \
    -DOpenCV_DIR=${OpenCV_DIR} \
    -DPADDLE_LIB=${PADDLE_LIB_DIR} \

make
