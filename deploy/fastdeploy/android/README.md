English | [简体中文](README_CN.md)
## PaddleClas Android Demo Tutorial

Real-time image classification on Android. This demo is easy to use for everyone. For example, you can run your own trained model in the Demo.

## Prepare the Environment

1. Install Android Studio in your local environment. Refer to [Android Studio Official Website](https://developer.android.com/studio)  for detailed tutorial.
2. Prepare an Android phone and turn on the USB debug mode: `Settings -> Find developer options -> Open developer options and USB debug mode`

## Deployment steps

1. The target detection PaddleClas Demo  is located in the `fastdeploy/examples/vision/classification/paddleclas/android`
2. Open paddleclas/android project with Android Studio
3. Connect the phone to the computer, turn on USB debug mode and file transfer mode, and connect your phone to Android Studio (allow the phone to install software from USB)

<p align="center">
<img width="1280" alt="image" src="https://user-images.githubusercontent.com/31974251/197338597-2c9e1cf0-569b-49b9-a7fb-cdec71921af8.png">
</p>

> **Attention：**
>> If you encounter an NDK configuration error during import, compilation or running, open ` File > Project Structure > SDK Location` and change the path of SDK configured by the `Andriod SDK location`.

4. Click the Run button to automatically compile the APP and install it to the phone. (The process will automatically download the pre-compiled FastDeploy Android library and model files. Internet is required).
The final effect is as follows. Figure 1: Install the APP on the phone; Figure 2: The effect when opening the APP. It will automatically recognize and mark the objects in the image; Figure 3: APP setting option. Click setting in the upper right corner and modify your options.

  | APP Icon | APP Effect | APP Settings
  | ---     | --- | --- |
  | ![app_pic ](https://user-images.githubusercontent.com/14995488/203484427-83de2316-fd60-4baf-93b6-3755f9b5559d.jpg)   | ![app_res](https://user-images.githubusercontent.com/14995488/203494666-16528cb3-0ce2-48fc-9f9e-37da17b2c2f6.jpg) |  ![app_setup](https://user-images.githubusercontent.com/14995488/203484436-57fdd041-7dcc-4e0e-b6cb-43e5ac1e729b.jpg) |

## PaddleClasModel Java API Description  
- Model initialized API: The initialized API contains two ways: Firstly, initialize directly through the constructor. Secondly, initialize at the appropriate program node by calling the init function. PaddleClasModel initialization parameters are as follows.
  - modelFile: String. Model file path in paddle format, such as model.pdmodel
  - paramFile: String. Parameter file path in paddle format, such as model.pdiparams  
  - configFile: String. Preprocessing file for model inference, such as infer_cfg.yml  
  - labelFile: String. This optional parameter indicates the path of the label file and is used for visualization, such as imagenet1k_label_list.txt, each line containing one label
  - option: RuntimeOption. Optional parameter for model initialization. Default runtime options if not passing the parameter.  

```java
// Constructor: constructor w/o label file
public PaddleClasModel(); // An empty constructor, which can be initialized by calling init
public PaddleClasModel(String modelFile, String paramsFile, String configFile);
public PaddleClasModel(String modelFile, String paramsFile, String configFile, String labelFile);
public PaddleClasModel(String modelFile, String paramsFile, String configFile, RuntimeOption option);
public PaddleClasModel(String modelFile, String paramsFile, String configFile, String labelFile, RuntimeOption option);
// Call init manually for initialization: call init manually w/o label file
public boolean init(String modelFile, String paramsFile, String configFile, RuntimeOption option);
public boolean init(String modelFile, String paramsFile, String configFile, String labelFile, RuntimeOption option);
```  
- Model Prediction API: The Model Prediction API contains an API for direct prediction and an API for visualization. In direct prediction, we do not save the image and render the result on Bitmap. Instead, we merely predict the inference result. For prediction and visualization, the results are both predicted and visualized, the visualized images are saved to the specified path, and the visualized results are rendered in Bitmap (Now Bitmap in ARGB8888 format is supported). Afterward, the Bitmap can be displayed on the camera.
```java
// Direct prediction: No image saving and no result rendering to Bitmap
public ClassifyResult predict(Bitmap ARGB8888Bitmap)；
// Prediction and visualization: Predict and visualize the results, save the visualized image to the specified path, and render the visualized results on Bitmap
public ClassifyResult predict(Bitmap ARGB8888Bitmap, String savedImagePath, float scoreThreshold)
```
- Model resource release API: Call release() API to release model resources. Return true for successful release and false for failure; call initialized() to determine whether the model was initialized successfully, with true indicating successful initialization and false indicating failure.
```java
public boolean release(); // Realise native resources
public boolean initialized(); // Check if initialization is successful
```
- RuntimeOption settings  
```java  
public void enableLiteFp16(); // Enable fp16 accuracy inference
public void disableLiteFP16(); // Disable fp16 accuracy inference
public void setCpuThreadNum(int threadNum); // Set thread numbers
public void setLitePowerMode(LitePowerMode mode);  // Set power mode
public void setLitePowerMode(String modeStr);  // Set power mode through character string
public void enableRecordTimeOfRuntime();  // Whether the print model is time-consuming
```

- Model ClassifyResult
```java
public float[] mScores;  // [n]   Score
public int[] mLabelIds;  // [n]   Classification ID
public boolean initialized(); //  Whether the result is valid or not
```  

- Model Calling Example 1: Using Constructor or Default RuntimeOption
```java  
import java.nio.ByteBuffer;
import android.graphics.Bitmap;
import android.opengl.GLES20;

import com.baidu.paddle.fastdeploy.vision.ClassifyResult;
import com.baidu.paddle.fastdeploy.vision.classification.PaddleClasModel;

// Initialize the model
PaddleClasModel model = new PaddleClasModel("MobileNetV1_x0_25_infer/inference.pdmodel",
                                            "MobileNetV1_x0_25_infer/inference.pdiparams",
                                            "MobileNetV1_x0_25_infer/inference_cls.yml");

// Read the image: The following is merely the pseudo code to read the Bitmap
ByteBuffer pixelBuffer = ByteBuffer.allocate(width * height * 4);
GLES20.glReadPixels(0, 0, width, height, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, pixelBuffer);
Bitmap ARGB8888ImageBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
ARGB8888ImageBitmap.copyPixelsFromBuffer(pixelBuffer);

// Model inference
ClassifyResult result = model.predict(ARGB8888ImageBitmap);  

// Release model resources  
model.release();
```  

- Model calling example 2: Manually call init at the appropriate program node and self-define RuntimeOption
```java  
// import is as the above...
import com.baidu.paddle.fastdeploy.RuntimeOption;
import com.baidu.paddle.fastdeploy.LitePowerMode;
import com.baidu.paddle.fastdeploy.vision.ClassifyResult;
import com.baidu.paddle.fastdeploy.vision.classification.PaddleClasModel;
// Create an empty model
PaddleClasModel model = new PaddleClasModel();  
// Model path
String modelFile = "MobileNetV1_x0_25_infer/inference.pdmodel";
String paramFile = "MobileNetV1_x0_25_infer/inference.pdiparams";
String configFile = "MobileNetV1_x0_25_infer/inference_cls.yml";
// Specify RuntimeOption
RuntimeOption option = new RuntimeOption();
option.setCpuThreadNum(2);
option.setLitePowerMode(LitePowerMode.LITE_POWER_HIGH);
option.enableRecordTimeOfRuntime();
option.enableLiteFp16();
// Use init function for initialization  
model.init(modelFile, paramFile, configFile, option);
// Bitmap reading, model prediction, and resource release are as above ...
```
Refer to [MainActivity](./app/src/main/java/com/baidu/paddle/fastdeploy/app/examples/classification/ClassificationMainActivity.java) for more information

## Replace FastDeploy Prediction Library and Models
It’s simple to replace the FastDeploy prediction library and models. The prediction library is located at `app/libs/fastdeploy-android-xxx-shared`, where `xxx` represents the version of your prediction library. The models are located at `app/src/main/assets/models/MobileNetV1_x0_25_infer`.  
- Steps to replace FastDeploy prediction library:
  - Download or compile the latest FastDeploy Android SDK, unzip and place it in the `app/libs`;  
  - Modify the default value of the model path in `app/src/main/cpp/CMakeLists.txt` and to the prediction library path you download or compile. For example,  
```cmake  
set(FastDeploy_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../libs/fastdeploy-android-xxx-shared")
```
- Steps to replace PaddleClas models:
  - Put your PaddleClas model in `app/src/main/assets/models`;
  - Modify the default value of the model path in `app/src/main/res/values/strings.xml`. For example,
```xml
<!-- Change this path to your model, such as models/MobileNetV2_x0_25_infer -->
<string name="CLASSIFICATION_MODEL_DIR_DEFAULT">models/MobileNetV1_x0_25_infer</string>  
<string name="CLASSIFICATION_LABEL_PATH_DEFAULT">labels/imagenet1k_label_list.txt</string>
```  

## More Reference Documents
For more FastDeploy Java API documentes and how to access FastDeploy C++ API via JNI, refer to:
- [Use FastDeploy Java SDK in Android](../../../../../java/android/)
- [Use FastDeploy C++ SDK in Android](../../../../../docs/en/faq/use_cpp_sdk_on_android.md)  
