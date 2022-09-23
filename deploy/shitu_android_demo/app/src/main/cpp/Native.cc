// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "Native.h"
#include "Pipeline.h"
#include <android/bitmap.h>

#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     Java_com_baidu_paddle_lite_demo_pp_1shitu_Native
 * Method:    nativeInit
 * Signature:
 * (Ljava/lang/String;Ljava/lang/String;ILjava/lang/String;II[F[FF)J
 */
JNIEXPORT jlong JNICALL
Java_com_baidu_paddle_lite_demo_pp_1shitu_Native_nativeInit(
    JNIEnv *env, jclass thiz, jstring jDetModelDir, jstring jRecModelDir,
    jstring jLabelPath, jstring jIndexPath, jlongArray jDetInputShape,
    jlongArray jRecInputShape, jint cpuThreadNum, jint WarmUp, jint Repeats,
    jint topk, jboolean jaddGallery, jstring cpu_power) {
  std::string det_model_path = jstring_to_cpp_string(env, jDetModelDir);
  std::string rec_model_path = jstring_to_cpp_string(env, jRecModelDir);
  std::string label_path = jstring_to_cpp_string(env, jLabelPath);
  std::string index_path = jstring_to_cpp_string(env, jIndexPath);
  bool add_gallery = jaddGallery;
  const std::string cpu_mode = jstring_to_cpp_string(env, cpu_power);
  std::vector<int64_t> det_input_shape =
      jlongarray_to_int64_vector(env, jDetInputShape);
  std::vector<int64_t> rec_input_shape =
      jlongarray_to_int64_vector(env, jRecInputShape);
  std::vector<int> det_input_shape_int;
  std::vector<int> rec_input_shape_int;
  for (auto &tmp : det_input_shape)
    det_input_shape_int.emplace_back(static_cast<int>(tmp));
  for (auto &tmp : rec_input_shape)
    rec_input_shape_int.emplace_back(static_cast<int>(tmp));
  return reinterpret_cast<jlong>(
      new PipeLine(det_model_path, rec_model_path, label_path, index_path,
                   det_input_shape_int, rec_input_shape_int, cpuThreadNum,
                   WarmUp, Repeats, topk, add_gallery, cpu_mode));
}

/*
 * Class:     Java_com_baidu_paddle_lite_demo_pp_1shitu_Native
 * Method:    nativeRelease
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_lite_demo_pp_1shitu_Native_nativeRelease(JNIEnv *env,
                                                               jclass thiz,
                                                               jlong ctx) {
  if (ctx == 0) {
    return JNI_FALSE;
  }
  auto *pipeline = reinterpret_cast<PipeLine *>(ctx);
  delete pipeline;
  return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_lite_demo_pp_1shitu_Native_nativesetAddGallery(
    JNIEnv *env, jclass thiz, jlong ctx, jboolean flag) {
  if (ctx == 0) {
    return JNI_FALSE;
  }
  auto *pipeline = reinterpret_cast<PipeLine *>(ctx);
  pipeline->set_add_gallery(flag);
  return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_lite_demo_pp_1shitu_Native_nativeclearGallery(JNIEnv *env,
                                                                    jclass thiz,
                                                                    jlong ctx) {
  if (ctx == 0) {
    return JNI_FALSE;
  }
  auto *pipeline = reinterpret_cast<PipeLine *>(ctx);
  pipeline->ClearFeature();
  return JNI_TRUE;
}

/*
 * Class:     Java_com_baidu_paddle_lite_demo_pp_1shitu_Native
 * Method:    nativeProcess
 * Signature: (JIIIILjava/lang/String;)Z
 */
JNIEXPORT jstring JNICALL
Java_com_baidu_paddle_lite_demo_pp_1shitu_Native_nativeProcess(
    JNIEnv *env, jclass thiz, jlong ctx, jobject jARGB8888ImageBitmap,
    jstring jlabel_name) {
  if (ctx == 0) {
    return JNI_FALSE;
  }

  // Convert the android bitmap(ARGB8888) to the OpenCV RGBA image. Actually,
  // the data layout of AGRB8888 is R, G, B, A, it's the same as CV RGBA image,
  // so it is unnecessary to do the conversion of color format, check
  // https://developer.android.com/reference/android/graphics/Bitmap.Config#ARGB_8888
  // to get the more details about Bitmap.Config.ARGB8888
  auto t = GetCurrentTime();
  void *bitmapPixels;
  AndroidBitmapInfo bitmapInfo;
  if (AndroidBitmap_getInfo(env, jARGB8888ImageBitmap, &bitmapInfo) < 0) {
    LOGE("Invoke AndroidBitmap_getInfo() failed!");
    return JNI_FALSE;
  }
  if (bitmapInfo.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
    LOGE("Only Bitmap.Config.ARGB8888 color format is supported!");
    return JNI_FALSE;
  }
  if (AndroidBitmap_lockPixels(env, jARGB8888ImageBitmap, &bitmapPixels) < 0) {
    LOGE("Invoke AndroidBitmap_lockPixels() failed!");
    return JNI_FALSE;
  }
  cv::Mat bmpImage(bitmapInfo.height, bitmapInfo.width, CV_8UC4, bitmapPixels);
  cv::Mat rgbaImage;
  std::string label_name = jstring_to_cpp_string(env, jlabel_name);
  bmpImage.copyTo(rgbaImage);
  if (AndroidBitmap_unlockPixels(env, jARGB8888ImageBitmap) < 0) {
    LOGE("Invoke AndroidBitmap_unlockPixels() failed!");
    return JNI_FALSE;
  }
  LOGD("Read from bitmap costs %f ms", GetElapsedTime(t));

  auto *pipeline = reinterpret_cast<PipeLine *>(ctx);

  std::vector<cv::Mat> input_mat;
  std::vector<ObjectResult> out_object;
  cv::Mat rgb_input;
  cv::cvtColor(rgbaImage, rgb_input, cv::COLOR_RGBA2RGB);
  input_mat.emplace_back(rgb_input);
  std::string res_str = pipeline->run(input_mat, out_object, 1, label_name);
  bool modified = res_str.empty();
  if (!modified) {
    cv::Mat res_img;
    cv::cvtColor(input_mat[0], res_img, cv::COLOR_RGB2RGBA);
    // Convert the OpenCV RGBA image to the android bitmap(ARGB8888)
    if (res_img.type() != CV_8UC4) {
      LOGE("Only CV_8UC4 color format is supported!");
      return JNI_FALSE;
    }
    t = GetCurrentTime();
    if (AndroidBitmap_lockPixels(env, jARGB8888ImageBitmap, &bitmapPixels) <
        0) {
      LOGE("Invoke AndroidBitmap_lockPixels() failed!");
      return JNI_FALSE;
    }
    cv::Mat bmpImage(bitmapInfo.height, bitmapInfo.width, CV_8UC4,
                     bitmapPixels);
    res_img.copyTo(bmpImage);
    if (AndroidBitmap_unlockPixels(env, jARGB8888ImageBitmap) < 0) {
      LOGE("Invoke AndroidBitmap_unlockPixels() failed!");
      return JNI_FALSE;
    }
    LOGD("Write to bitmap costs %f ms", GetElapsedTime(t));
  }
  return cpp_string_to_jstring(env, res_str);
}

#ifdef __cplusplus
}
#endif

extern "C" JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_lite_demo_pp_1shitu_Native_nativesaveIndex(
    JNIEnv *env, jclass clazz, jlong ctx, jstring jsave_file_name) {
  // TODO: implement nativesaveIndex()
  if (ctx == 0) {
    return JNI_FALSE;
  }
  auto *pipeline = reinterpret_cast<PipeLine *>(ctx);
  std::string save_file_name = jstring_to_cpp_string(env, jsave_file_name);
  pipeline->SaveIndex(save_file_name);
  return JNI_TRUE;
}
extern "C" JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_lite_demo_pp_1shitu_Native_nativeloadIndex(
    JNIEnv *env, jclass clazz, jlong ctx, jstring jload_file_name) {
  // TODO: implement nativeloadIndex()
  if (ctx == 0) {
    return JNI_FALSE;
  }
  auto *pipeline = reinterpret_cast<PipeLine *>(ctx);
  std::string load_file_name = jstring_to_cpp_string(env, jload_file_name);
  bool load_flag = pipeline->LoadIndex(load_file_name);
  return JNI_TRUE && load_flag;
}
extern "C" JNIEXPORT jstring JNICALL
Java_com_baidu_paddle_lite_demo_pp_1shitu_Native_nativegetClassname(
    JNIEnv *env, jclass clazz, jlong ctx) {
  // TODO: implement nativegetClassname()
  if (ctx == 0) {
    return JNI_FALSE;
  }
  auto *pipeline = reinterpret_cast<PipeLine *>(ctx);
  std::string class_name_content = pipeline->GetLabelList();
  return cpp_string_to_jstring(env, class_name_content);
}