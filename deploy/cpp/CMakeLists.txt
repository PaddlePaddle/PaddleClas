PROJECT(clas_system CXX C)
CMAKE_MINIMUM_REQUIRED(VERSION 3.14)

OPTION(WITH_MKL        "Compile demo with MKL/OpenBlas support, default use MKL."       ON)
OPTION(WITH_GPU        "Compile demo with GPU/CPU, default use CPU."                    OFF)
OPTION(WITH_STATIC_LIB "Compile demo with static/shared library, default use static."   ON)
OPTION(WITH_TENSORRT "Compile demo with TensorRT."   OFF)
OPTION(WHOLE_CHAIN "Whole chain testing switch" OFF)

SET(PADDLE_LIB "" CACHE PATH "The directory of Paddle Paddle Labrary.")
SET(OpenCV_DIR "" CACHE PATH "The directory of OpenCV Library")
SET(CUDA_LIB "" CACHE PATH "The root directory of CUDA Library")
SET(CUDNN_LIB "" CACHE PATH "The root directory of cuDNN Library")
SET(TENSORRT_LIB "" CACHE PATH "The root directory of TensorRT library")

# TODO(gaotingquan): is this necessary?
if(NOT WITH_STATIC_LIB)
  add_definitions("-DPADDLE_WITH_SHARED_LIB")
else()
  # PD_INFER_DECL is mainly used to set the dllimport/dllexport attribute in dynamic library mode.
  # Set it to empty in static library mode to avoid compilation issues.
  add_definitions("/DPD_INFER_DECL=")
endif()

MACRO(safe_set_static_flag)
    FOREACH(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      IF(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      ENDIF()
    ENDFOREACH()
ENDMACRO()

IF(NOT DEFINED OpenCV_DIR)
    MESSAGE(FATAL_ERROR "Please set OpenCV_DIR with -DOpenCV_DIR=/path/opencv.")
ENDIF()

IF(NOT DEFINED PADDLE_LIB)
  MESSAGE(FATAL_ERROR "Please set PADDLE_LIB with -DPADDLE_LIB=/path/paddle/lib.")
ENDIF()

INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}/include")

FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
SET(DEPS ${OpenCV_LIBS})

SET(PADDLE_LIB_THIRD_PARTY_PATH "${PADDLE_LIB}/third_party/install")
INCLUDE_DIRECTORIES("${PADDLE_LIB}/paddle/include")
INCLUDE_DIRECTORIES("${PADDLE_LIB_THIRD_PARTY_PATH}/protobuf/include")
INCLUDE_DIRECTORIES("${PADDLE_LIB_THIRD_PARTY_PATH}/glog/include")
INCLUDE_DIRECTORIES("${PADDLE_LIB_THIRD_PARTY_PATH}/gflags/include")
INCLUDE_DIRECTORIES("${PADDLE_LIB_THIRD_PARTY_PATH}/xxhash/include")
INCLUDE_DIRECTORIES("${PADDLE_LIB_THIRD_PARTY_PATH}/cryptopp/include")

LINK_DIRECTORIES("${PADDLE_LIB}/paddle/lib")
LINK_DIRECTORIES("${PADDLE_LIB_THIRD_PARTY_PATH}/protobuf/lib")
LINK_DIRECTORIES("${PADDLE_LIB_THIRD_PARTY_PATH}/glog/lib/")
LINK_DIRECTORIES("${PADDLE_LIB_THIRD_PARTY_PATH}/gflags/lib")
LINK_DIRECTORIES("${PADDLE_LIB_THIRD_PARTY_PATH}/xxhash/lib")
LINK_DIRECTORIES("${PADDLE_LIB_THIRD_PARTY_PATH}/cryptopp/lib")

IF(WITH_GPU)
    IF (NOT DEFINED CUDA_LIB OR ${CUDA_LIB} STREQUAL "")
        MESSAGE(FATAL_ERROR "If use GPU, please set CUDA_LIB with -DCUDA_LIB=/path/cuda-8.0/lib64.")
    ENDIF()

    IF(WIN32)
      SET(DEPS ${DEPS} ${CUDA_LIB}/cudart${CMAKE_STATIC_LIBRARY_SUFFIX} )
      SET(DEPS ${DEPS} ${CUDA_LIB}/cublas${CMAKE_STATIC_LIBRARY_SUFFIX} )
      SET(DEPS ${DEPS} ${CUDNN_LIB}/cudnn${CMAKE_STATIC_LIBRARY_SUFFIX})
    ELSE()
      IF(NOT DEFINED CUDNN_LIB OR ${CUDNN_LIB} STREQUAL "")
        MESSAGE(FATAL_ERROR "If use GPU, please set CUDNN_LIB with -DCUDNN_LIB=/path/cudnn_v7.4/cuda/lib64.")
      ENDIF()
      SET(DEPS ${DEPS} ${CUDA_LIB}/libcudart${CMAKE_SHARED_LIBRARY_SUFFIX})
      SET(DEPS ${DEPS} ${CUDNN_LIB}/libcudnn${CMAKE_SHARED_LIBRARY_SUFFIX})

      IF(WITH_TENSORRT)
        IF(NOT DEFINED TENSORRT_LIB OR ${TENSORRT_LIB} STREQUAL "")
          MESSAGE(FATAL_ERROR "If use TensorRT, please set TENSORRT_LIB with -DTENSORRT_LIB=/path/to/tensorrt/.")
        ENDIF()
        INCLUDE_DIRECTORIES("${TENSORRT_LIB}/include")
        LINK_DIRECTORIES("${TENSORRT_LIB}/lib")
        SET(DEPS ${DEPS} ${TENSORRT_LIB}/lib/libnvinfer${CMAKE_SHARED_LIBRARY_SUFFIX})
        SET(DEPS ${DEPS} ${TENSORRT_LIB}/lib/libnvinfer_plugin${CMAKE_SHARED_LIBRARY_SUFFIX})
      ENDIF()
    ENDIF()

    # GPU paddle_inference library is based on MKL.
    SET(WITH_MKL ON)
ENDIF()

IF(WITH_MKL)
  IF(NOT EXISTS ${PADDLE_LIB_THIRD_PARTY_PATH}/mklml)
    MESSAGE(FATAL_ERROR "If use MKL, please download the paddle_inference library with MKL.")
  ENDIF()
  INCLUDE_DIRECTORIES("${PADDLE_LIB_THIRD_PARTY_PATH}/mklml/include")

  IF(WIN32)
    SET(MATH_LIB ${PADDLE_LIB_THIRD_PARTY_PATH}/mklml/lib/mklml.lib
            ${PADDLE_LIB_THIRD_PARTY_PATH}/mklml/lib/libiomp5md.lib)
  ELSE()
    SET(MATH_LIB ${PADDLE_LIB_THIRD_PARTY_PATH}/mklml/lib/libmklml_intel${CMAKE_SHARED_LIBRARY_SUFFIX}
            ${PADDLE_LIB_THIRD_PARTY_PATH}/mklml/lib/libiomp5${CMAKE_SHARED_LIBRARY_SUFFIX})
    EXECUTE_PROCESS(COMMAND cp -r ${PADDLE_LIB_THIRD_PARTY_PATH}/mklml/lib/libmklml_intel${CMAKE_SHARED_LIBRARY_SUFFIX} /usr/lib)
  ENDIF()

  SET(MKLDNN_PATH "${PADDLE_LIB_THIRD_PARTY_PATH}/mkldnn")
  IF(EXISTS ${MKLDNN_PATH})
    INCLUDE_DIRECTORIES("${MKLDNN_PATH}/include")
    IF (WIN32)
      SET(MKLDNN_LIB ${MKLDNN_PATH}/lib/mkldnn.lib)
    ELSE ()
      SET(MKLDNN_LIB ${MKLDNN_PATH}/lib/libmkldnn.so.0)
    ENDIF ()
    SET(DEPS ${DEPS} ${MKLDNN_LIB})
  ENDIF()
ELSE()
  IF(NOT EXISTS ${PADDLE_LIB_THIRD_PARTY_PATH}/openblas)
    MESSAGE(FATAL_ERROR "If use OpenBLAS, please download the paddle_inference library with OpenBLAS.")
  ENDIF()
  IF (WIN32)
    SET(MATH_LIB ${PADDLE_LIB_THIRD_PARTY_PATH}/openblas/lib/openblas${CMAKE_STATIC_LIBRARY_SUFFIX})
  ELSE ()
    SET(MATH_LIB ${PADDLE_LIB_THIRD_PARTY_PATH}/openblas/lib/libopenblas${CMAKE_STATIC_LIBRARY_SUFFIX})
  ENDIF ()
ENDIF()
SET(DEPS ${DEPS} ${MATH_LIB})

# for Win
IF(WIN32)
  IF(WITH_STATIC_LIB)
    SET(DEPS ${DEPS} ${PADDLE_LIB}/paddle/lib/libpaddle_inference${CMAKE_STATIC_LIBRARY_SUFFIX})
  ELSE()
    SET(DEPS ${DEPS} ${PADDLE_LIB}/paddle/lib/paddle_inference${CMAKE_SHARED_LIBRARY_SUFFIX})
  ENDIF()

  SET(DEPS ${DEPS} glog gflags_static libprotobuf xxhash cryptopp-static shlwapi.lib)

  ADD_DEFINITIONS("/DGOOGLE_GLOG_DLL_DECL=")
  #TODO(gaotingquan): Is it necessary to set MSVC_STATIC_CRT?
  option(MSVC_STATIC_CRT "use static C Runtime library by default" ON)
  if(MSVC_STATIC_CRT)
    if(WITH_MKL)
      set(FLAG_OPENMP "/openmp")
    endif()
    set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} /bigobj /MTd ${FLAG_OPENMP}")
    set(CMAKE_C_FLAGS_RELEASE  "${CMAKE_C_FLAGS_RELEASE} /bigobj /MT ${FLAG_OPENMP}")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} /bigobj /MTd ${FLAG_OPENMP}")
    set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_CXX_FLAGS_RELEASE} /bigobj /MT ${FLAG_OPENMP}")
    #TODO(gaotingquan): where should safe_set_static_flag run?
    #safe_set_static_flag()
    if(WITH_STATIC_LIB)
      safe_set_static_flag()
      add_definitions(-DSTATIC_LIB)
    endif()
  endif()

  # SET(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} /bigobj /MTd")
  # SET(CMAKE_C_FLAGS_RELEASE  "${CMAKE_C_FLAGS_RELEASE} /bigobj /MT")
  # SET(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} /bigobj /MTd")
  # SET(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_CXX_FLAGS_RELEASE} /bigobj /MT")
  # IF (WITH_STATIC_LIB)
  #     safe_set_static_flag()
  #     ADD_DEFINITIONS(-DSTATIC_LIB)
  # ENDIF()

# for Linux
ELSE()
  IF(WITH_STATIC_LIB)
    SET(DEPS ${DEPS} ${PADDLE_LIB}/paddle/lib/libpaddle_inference${CMAKE_STATIC_LIBRARY_SUFFIX})
  ELSE()
    SET(DEPS ${DEPS} ${PADDLE_LIB}/paddle/lib/libpaddle_inference${CMAKE_SHARED_LIBRARY_SUFFIX})
  ENDIF()

  SET(DEPS ${DEPS} glog gflags protobuf xxhash cryptopp)
  SET(EXTERNAL_LIB "-ldl -lrt -lgomp -lz -lm -lpthread")
  SET(DEPS ${DEPS} ${EXTERNAL_LIB})

  IF(WITH_MKL)
    SET(FLAG_OPENMP "-fopenmp")
  ENDIF()
  # TODO(gaotingquan):
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ${FLAG_OPENMP}")
  # SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -o3 -std=c++11")
  # SET(CMAKE_STATIC_LIBRARY_PREFIX "")
ENDIF()
MESSAGE(STATUS "FLAGS: ${CMAKE_CXX_FLAGS}")

SET(DEPS ${DEPS} ${OpenCV_LIBS})

IF(WHOLE_CHAIN)
  INCLUDE(FetchContent)
  INCLUDE(external-cmake/auto-log.cmake)
  INCLUDE_DIRECTORIES(${FETCHCONTENT_BASE_DIR}/extern_autolog-src)
  ADD_DEFINITIONS(-DWHOLE_CHAIN)
ENDIF()

ADD_EXECUTABLE(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/test/main.cpp)

IF(DEFINED CONFIG_LIB)
  GET_FILENAME_COMPONENT(CONFIG_LIB ${CONFIG_LIB} ABSOLUTE)
ENDIF()
IF(DEFINED CLS_LIB)
  GET_FILENAME_COMPONENT(CLS_LIB ${CLS_LIB} ABSOLUTE)
ENDIF()

IF(NOT EXISTS "${CONFIG_LIB}" OR NOT EXISTS "${CLS_LIB}")
  MESSAGE(WARNING "The CONFIG_LIB or CLS_LIB is not set.")
  ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/lib)
  TARGET_LINK_LIBRARIES(${PROJECT_NAME} config cls ${DEPS})
ELSE()
  MESSAGE(WARNING "Use the existing CONFIG_LIB and existing CLS_LIB.")
  TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${CONFIG_LIB} ${CLS_LIB} ${DEPS})
ENDIF()

IF (WIN32 AND WITH_MKL)
    ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_LIB_THIRD_PARTY_PATH}/mklml/lib/mklml.dll ./mklml.dll
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_LIB_THIRD_PARTY_PATH}/mklml/lib/libiomp5md.dll ./libiomp5md.dll
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_LIB_THIRD_PARTY_PATH}/mkldnn/lib/mkldnn.dll ./mkldnn.dll
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_LIB_THIRD_PARTY_PATH}/mklml/lib/mklml.dll ./release/mklml.dll
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_LIB_THIRD_PARTY_PATH}/mklml/lib/libiomp5md.dll ./release/libiomp5md.dll
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_LIB_THIRD_PARTY_PATH}/mkldnn/lib/mkldnn.dll ./release/mkldnn.dll
    )
ENDIF()
