ARM_ABI = arm8
export ARM_ABI

LITE_ROOT=./inference_lite_lib.android.armv8

include ${LITE_ROOT}/demo/cxx/Makefile.def

THIRD_PARTY_DIR=${LITE_ROOT}/third_party

OPENCV_VERSION=opencv4.1.0

ifeq (${ARM_ABI}, arm8)
    ARM_PATH=arm64-v8a
endif
ifeq (${ARM_ABI}, arm7)
    ARM_PATH=armeabi-v7a
endif

OPENCV_LIBS = ${THIRD_PARTY_DIR}/${OPENCV_VERSION}/${ARM_PATH}/libs/libopencv_imgcodecs.a \
              ${THIRD_PARTY_DIR}/${OPENCV_VERSION}/${ARM_PATH}/libs/libopencv_imgproc.a \
              ${THIRD_PARTY_DIR}/${OPENCV_VERSION}/${ARM_PATH}/libs/libopencv_core.a \
              ${THIRD_PARTY_DIR}/${OPENCV_VERSION}/${ARM_PATH}/3rdparty/libs/libtegra_hal.a \
              ${THIRD_PARTY_DIR}/${OPENCV_VERSION}/${ARM_PATH}/3rdparty/libs/liblibjpeg-turbo.a \
              ${THIRD_PARTY_DIR}/${OPENCV_VERSION}/${ARM_PATH}/3rdparty/libs/liblibwebp.a \
              ${THIRD_PARTY_DIR}/${OPENCV_VERSION}/${ARM_PATH}/3rdparty/libs/liblibpng.a \
              ${THIRD_PARTY_DIR}/${OPENCV_VERSION}/${ARM_PATH}/3rdparty/libs/liblibjasper.a \
              ${THIRD_PARTY_DIR}/${OPENCV_VERSION}/${ARM_PATH}/3rdparty/libs/liblibtiff.a \
              ${THIRD_PARTY_DIR}/${OPENCV_VERSION}/${ARM_PATH}/3rdparty/libs/libIlmImf.a \
              ${THIRD_PARTY_DIR}/${OPENCV_VERSION}/${ARM_PATH}/3rdparty/libs/libtbb.a \
              ${THIRD_PARTY_DIR}/${OPENCV_VERSION}/${ARM_PATH}/3rdparty/libs/libcpufeatures.a

OPENCV_INCLUDE = -I${LITE_ROOT}/third_party/${OPENCV_VERSION}/${ARM_PATH}/include

CXX_INCLUDES = $(INCLUDES) ${OPENCV_INCLUDE} -I$(LITE_ROOT)/cxx/include

CXX_LIBS = ${OPENCV_LIBS} -L$(LITE_ROOT)/cxx/lib/ -lpaddle_light_api_shared $(SYSTEM_LIBS)

###############################################################
# How to use one of static libaray:                           #
#  `libpaddle_api_full_bundled.a`                             #
#  `libpaddle_api_light_bundled.a`                            #
###############################################################
# Note: default use lite's shared library.                    #
###############################################################
# 1. Comment above line using `libpaddle_light_api_shared.so`
# 2. Undo comment below line using `libpaddle_api_light_bundled.a`

#CXX_LIBS = $(LITE_ROOT)/cxx/lib/libpaddle_api_light_bundled.a $(SYSTEM_LIBS)

clas_system: fetch_opencv clas_system.o
	$(CC) $(SYSROOT_LINK) $(CXXFLAGS_LINK) clas_system.o -o clas_system  $(CXX_LIBS) $(LDFLAGS)

clas_system.o: image_classfication.cpp
	$(CC) $(SYSROOT_COMPLILE) $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o clas_system.o -c image_classfication.cpp

fetch_opencv:
	@ test -d ${THIRD_PARTY_DIR} ||  mkdir ${THIRD_PARTY_DIR}
	@ test -e ${THIRD_PARTY_DIR}/${OPENCV_VERSION}.tar.gz || \
      (echo "fetch opencv libs" && \
      wget -P ${THIRD_PARTY_DIR} https://paddle-inference-dist.bj.bcebos.com/${OPENCV_VERSION}.tar.gz)
	@ test -d ${THIRD_PARTY_DIR}/${OPENCV_VERSION} || \
      tar -zxvf ${THIRD_PARTY_DIR}/${OPENCV_VERSION}.tar.gz -C ${THIRD_PARTY_DIR}


.PHONY: clean
clean:
	rm -f clas_system.o
	rm -f clas_system
