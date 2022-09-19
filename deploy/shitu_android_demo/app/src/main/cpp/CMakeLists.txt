# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.
#project(ShituDemo)

cmake_minimum_required(VERSION 3.4.1)

# Creates and names a library, sets it as either STATIC or SHARED, and provides
# the relative paths to its source code. You can define multiple libraries, and
# CMake builds them for you. Gradle automatically packages shared libraries with
# your APK.

set(PaddleLite_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../PaddleLite")
include_directories(${PaddleLite_DIR}/cxx/include)

set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../OpenCV/sdk/native/jni")
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
include_directories(${OpenCV_INCLUDE_DIRS})

#set(PaddleLite_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../PaddleLite")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../PaddleLite/src/main/cpp/include/faiss)
set(target faiss)

set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -ffast-math -Ofast -Os -DNDEBUG -fexceptions -fomit-frame-pointer -fno-asynchronous-unwind-tables -fno-unwind-tables"
)
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -fvisibility=hidden -fvisibility-inlines-hidden -fdata-sections -ffunction-sections"
)
set(CMAKE_SHARED_LINKER_FLAGS
    "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections -Wl,-z,nocopyreloc")

add_library(
  # Sets the name of the library.
  Native
  # Sets the library as a shared library.
  SHARED
  # Provides a relative path to your source file(s).
  Native.cc Pipeline.cc Utils.cc ObjectDetector.cc FeatureExtractor.cc VectorSearch.cc)

find_library(
  # Sets the name of the path variable.
  log-lib
  # Specifies the name of the NDK library that you want CMake to locate.
  log)

add_library(
  # Sets the name of the library.
  paddle_light_api_shared
  # Sets the library as a shared library.
  SHARED
  # Provides a relative path to your source file(s).
  IMPORTED)

add_library(
  # Sets the name of the library.
  faiss
  STATIC
  IMPORTED)

set_target_properties(
  # Specifies the target library.
  paddle_light_api_shared
  # Specifies the parameter you want to define.
  PROPERTIES
    IMPORTED_LOCATION
    ${PaddleLite_DIR}/cxx/libs/${ANDROID_ABI}/libpaddle_light_api_shared.so
    # Provides the path to the library you want to import.
)

# if there libfaiss.a not exist, will download it automatically
IF(NOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/main/jniLibs/arm64-v8a/libfaiss.a)
    message(STATUS "Downloading ${OCI_LIB_ZIP_NAME} to ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/main/jniLibs/arm64-v8a/")
    FILE(DOWNLOAD https://paddle-imagenet-models-name.bj.bcebos.com/demos/lib/libfaiss.a
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/main/jniLibs/arm64-v8a/libfaiss.a
            TIMEOUT ${DOWNLOAD_OCI_LIB_TIMEOUT}
            STATUS ERR
            SHOW_PROGRESS)
ENDIF()

set_target_properties(
    faiss
    PROPERTIES
    IMPORTED_LOCATION
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/main/jniLibs/arm64-v8a/libfaiss.a
)

# Specifies libraries CMake should link to your target library. You can link
# multiple libraries, such as libraries you define in this build script,
# prebuilt third-party libraries, or system libraries.

target_link_libraries(
  # Specifies the target library.
  Native
  paddle_light_api_shared
  jnigraphics
  ${OpenCV_LIBS}
  GLESv2
  EGL
  ${log-lib}
  faiss
)
