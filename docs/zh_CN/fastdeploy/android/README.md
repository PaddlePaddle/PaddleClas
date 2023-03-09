# PaddleClas Android Demo 使用文档

在 Android 上实现实时的PaddleClas图像分类功能，此 Demo 有很好的的易用性和开放性，如在 Demo 中跑自己训练好的模型等。

## 环境准备

1. 在本地环境安装好 Android Studio 工具，详细安装方法请见[Android Stuido 官网](https://developer.android.com/studio)。
2. 准备一部 Android 手机，并开启 USB 调试模式。开启方法: `手机设置 -> 查找开发者选项 -> 打开开发者选项和 USB 调试模式`

## 部署步骤

1. 用 Android Studio 打开 paddleclas/android 工程
2. 手机连接电脑，打开 USB 调试和文件传输模式，并在 Android Studio 上连接自己的手机设备（手机需要开启允许从 USB 安装软件权限）

<p align="center">
<img width="1280" alt="image" src="https://user-images.githubusercontent.com/31974251/197338597-2c9e1cf0-569b-49b9-a7fb-cdec71921af8.png">
</p>

> **注意：**
>> 如果您在导入项目、编译或者运行过程中遇到 NDK 配置错误的提示，请打开 ` File > Project Structure > SDK Location`，修改 `Andriod SDK location` 为您本机配置的 SDK 所在路径。

4. 点击 Run 按钮，自动编译 APP 并安装到手机。(该过程会自动下载预编译的 FastDeploy Android 库，需要联网)
成功后效果如下，图一：APP 安装到手机；图二： APP 打开后的效果，会自动识别图片中的物体并标记；图三：APP设置选项，点击右上角的设置图片，可以设置不同选项进行体验。

  | APP 图标 | APP 效果 | APP设置项
  | ---     | --- | --- |
  | ![app_pic ](https://user-images.githubusercontent.com/14995488/203484427-83de2316-fd60-4baf-93b6-3755f9b5559d.jpg)   | ![app_res](https://user-images.githubusercontent.com/14995488/203494666-16528cb3-0ce2-48fc-9f9e-37da17b2c2f6.jpg) |  ![app_setup](https://user-images.githubusercontent.com/14995488/203484436-57fdd041-7dcc-4e0e-b6cb-43e5ac1e729b.jpg) |

## PaddleClasModel Java API 说明  
- 模型初始化 API: 模型初始化API包含两种方式，方式一是通过构造函数直接初始化；方式二是，通过调用init函数，在合适的程序节点进行初始化。PaddleClasModel初始化参数说明如下：  
  - modelFile: String, paddle格式的模型文件路径，如 model.pdmodel
  - paramFile: String, paddle格式的参数文件路径，如 model.pdiparams  
  - configFile: String, 模型推理的预处理配置文件，如 infer_cfg.yml  
  - labelFile: String, 可选参数，表示label标签文件所在路径，用于可视化，如 imagenet1k_label_list.txt，每一行包含一个label  
  - option: RuntimeOption，可选参数，模型初始化option。如果不传入该参数则会使用默认的运行时选项。  

```java
// 构造函数: constructor w/o label file
public PaddleClasModel(); // 空构造函数，之后可以调用init初始化
public PaddleClasModel(String modelFile, String paramsFile, String configFile);
public PaddleClasModel(String modelFile, String paramsFile, String configFile, String labelFile);
public PaddleClasModel(String modelFile, String paramsFile, String configFile, RuntimeOption option);
public PaddleClasModel(String modelFile, String paramsFile, String configFile, String labelFile, RuntimeOption option);
// 手动调用init初始化: call init manually w/o label file
public boolean init(String modelFile, String paramsFile, String configFile, RuntimeOption option);
public boolean init(String modelFile, String paramsFile, String configFile, String labelFile, RuntimeOption option);
```  
- 模型预测 API：模型预测API包含直接预测的API以及带可视化功能的API。直接预测是指，不保存图片以及不渲染结果到Bitmap上，仅预测推理结果。预测并且可视化是指，预测结果以及可视化，并将可视化后的图片保存到指定的途径，以及将可视化结果渲染在Bitmap(目前支持ARGB8888格式的Bitmap), 后续可将该Bitmap在camera中进行显示。
```java
// 直接预测：不保存图片以及不渲染结果到Bitmap上
public ClassifyResult predict(Bitmap ARGB8888Bitmap)；
// 预测并且可视化：预测结果以及可视化，并将可视化后的图片保存到指定的途径，以及将可视化结果渲染在Bitmap上
public ClassifyResult predict(Bitmap ARGB8888Bitmap, String savedImagePath, float scoreThreshold)
```
- 模型资源释放 API：调用 release() API 可以释放模型资源，返回true表示释放成功，false表示失败；调用 initialized() 可以判断模型是否初始化成功，true表示初始化成功，false表示失败。
```java
public boolean release(); // 释放native资源  
public boolean initialized(); // 检查是否初始化成功
```
- RuntimeOption设置说明  
```java  
public void enableLiteFp16(); // 开启fp16精度推理
public void disableLiteFP16(); // 关闭fp16精度推理
public void setCpuThreadNum(int threadNum); // 设置线程数
public void setLitePowerMode(LitePowerMode mode);  // 设置能耗模式
public void setLitePowerMode(String modeStr);  // 通过字符串形式设置能耗模式
public void enableRecordTimeOfRuntime();  // 是否打印模型运行耗时
```

- 模型结果ClassifyResult说明  
```java
public float[] mScores;  // [n]   得分
public int[] mLabelIds;  // [n]   分类ID
public boolean initialized(); // 检测结果是否有效
```  

- 模型调用示例1：使用构造函数以及默认的RuntimeOption
```java  
import java.nio.ByteBuffer;
import android.graphics.Bitmap;
import android.opengl.GLES20;

import com.baidu.paddle.fastdeploy.vision.ClassifyResult;
import com.baidu.paddle.fastdeploy.vision.classification.PaddleClasModel;

// 初始化模型
PaddleClasModel model = new PaddleClasModel("MobileNetV1_x0_25_infer/inference.pdmodel",
                                            "MobileNetV1_x0_25_infer/inference.pdiparams",
                                            "MobileNetV1_x0_25_infer/inference_cls.yml");

// 读取图片: 以下仅为读取Bitmap的伪代码
ByteBuffer pixelBuffer = ByteBuffer.allocate(width * height * 4);
GLES20.glReadPixels(0, 0, width, height, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, pixelBuffer);
Bitmap ARGB8888ImageBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
ARGB8888ImageBitmap.copyPixelsFromBuffer(pixelBuffer);

// 模型推理
ClassifyResult result = model.predict(ARGB8888ImageBitmap);  

// 释放模型资源  
model.release();
```  

- 模型调用示例2: 在合适的程序节点，手动调用init，并自定义RuntimeOption
```java  
// import 同上 ...
import com.baidu.paddle.fastdeploy.RuntimeOption;
import com.baidu.paddle.fastdeploy.LitePowerMode;
import com.baidu.paddle.fastdeploy.vision.ClassifyResult;
import com.baidu.paddle.fastdeploy.vision.classification.PaddleClasModel;
// 新建空模型
PaddleClasModel model = new PaddleClasModel();  
// 模型路径
String modelFile = "MobileNetV1_x0_25_infer/inference.pdmodel";
String paramFile = "MobileNetV1_x0_25_infer/inference.pdiparams";
String configFile = "MobileNetV1_x0_25_infer/inference_cls.yml";
// 指定RuntimeOption
RuntimeOption option = new RuntimeOption();
option.setCpuThreadNum(2);
option.setLitePowerMode(LitePowerMode.LITE_POWER_HIGH);
option.enableRecordTimeOfRuntime();
option.enableLiteFp16();
// 使用init函数初始化  
model.init(modelFile, paramFile, configFile, option);
// Bitmap读取、模型预测、资源释放 同上 ...
```
更详细的用法请参考 [MainActivity](./app/src/main/java/com/baidu/paddle/fastdeploy/app/examples/classification/ClassificationMainActivity.java) 中的用法

## 替换 FastDeploy 预测库和模型  
替换FastDeploy预测库和模型的步骤非常简单。预测库所在的位置为 `app/libs/fastdeploy-android-xxx-shared`，其中 `xxx` 表示当前您使用的预测库版本号。模型所在的位置为，`app/src/main/assets/models/MobileNetV1_x0_25_infer`。  
- 替换FastDeploy预测库的步骤:
  - 下载或编译最新的FastDeploy Android预测库，解压缩后放在 `app/libs` 目录下;  
  - 修改 `app/src/main/cpp/CMakeLists.txt` 中的预测库路径，指向您下载或编译的预测库路径。如：  
```cmake  
set(FastDeploy_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../libs/fastdeploy-android-xxx-shared")
```
- 替换PaddleClas模型的步骤：
  - 将您的PaddleClas分类模型放在 `app/src/main/assets/models` 目录下；  
  - 修改 `app/src/main/res/values/strings.xml` 中模型路径的默认值，如：  
```xml
<!-- 将这个路径指修改成您的模型，如 models/MobileNetV2_x0_25_infer -->
<string name="CLASSIFICATION_MODEL_DIR_DEFAULT">models/MobileNetV1_x0_25_infer</string>  
<string name="CLASSIFICATION_LABEL_PATH_DEFAULT">labels/imagenet1k_label_list.txt</string>
```  

## 更多参考文档
如果您想知道更多的FastDeploy Java API文档以及如何通过JNI来接入FastDeploy C++ API感兴趣，可以参考以下内容:
- [在 Android 中使用 FastDeploy Java SDK](https://github.com/PaddlePaddle/FastDeploy/tree/develop/java/android)
- [在 Android 中使用 FastDeploy C++ SDK](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/use_cpp_sdk_on_android.md)  
