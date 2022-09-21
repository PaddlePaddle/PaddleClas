/*
 * Copyright (C) 2014 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.baidu.paddle.lite.demo.pp_shitu;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.AlertDialog;
import android.app.ProgressDialog;
import android.content.ContentResolver;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.ApplicationInfo;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Typeface;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Message;
import android.preference.PreferenceManager;
import android.provider.MediaStore;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.text.SpannableString;
import android.text.Spanned;
import android.text.style.StyleSpan;
import android.util.Log;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.baidu.paddle.lite.demo.common.Utils;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = MainActivity.class.getSimpleName();
    public static final int OPEN_QUERY_PHOTO_REQUEST_CODE = 0;
    public static final int TAKE_QUERY_PHOTO_REQUEST_CODE = 1;
    public static final int OPEN_GALLERY_PHOTO_REQUEST_CODE = 4;
    public static final int TAKE_GALLERY_PHOTO_REQUEST_CODE = 5;
    public static final int CLEAR_FEATURE_REQUEST_CODE = 6;
    public static final int REQUEST_LOAD_MODEL = 0;
    public static final int REQUEST_RUN_MODEL = 1;
    public static final int RESPONSE_LOAD_MODEL_SUCCESSED = 0;
    public static final int RESPONSE_LOAD_MODEL_FAILED = 1;
    public static final int RESPONSE_RUN_MODEL_SUCCESSED = 2;
    public static final int RESPONSE_RUN_MODEL_FAILED = 3;

    protected ProgressDialog pbLoadModel = null;
    protected ProgressDialog pbRunModel = null;
    protected Handler receiver = null; // Receive messages from worker thread
    protected Handler sender = null; // Send command to worker thread
    protected HandlerThread worker = null; // Worker thread to load&run model

    // UI components of image classification
    protected ImageView ivInputImage;
    protected TextView tvTop1Result;
    protected TextView tvInferenceTime;
    protected TextView tvSimilarity;
    protected TextView tvIndexName;
    protected TextView tv_description;

    //protected Switch mSwitch;

    // Model settings of image classification
    protected String modelPath = "";
    protected String labelPath = "";
    protected String indexPath = "";

    protected String imagePath = "";
    protected String DetModelPath = "";
    protected String RecModelPath = "";
    protected int cpuThreadNum = 1;
    protected EditText label_name;
    protected Button label_botton;
    protected Button cancel_botton;
    protected boolean add_gallery = false;
    protected int topk = 3;
    protected String cpuMode = "";
    protected long[] detinputShape = new long[]{};
    protected long[] recinputShape = new long[]{};
    protected boolean useGpu = false;
    protected Native predictor = new Native();

    ImageView mImage_add_query;
    ImageView mImage_take_query;
    ImageView mImage_add_gallery;
    ImageView mImage_take_gallery;
    ImageView mImage_save;
    ImageView mShow_index;
    View help;
    View reset;

    @SuppressLint("HandlerLeak")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Clear all setting items to avoid app crashing due to the incorrect settings
        SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this);
        SharedPreferences.Editor editor = sharedPreferences.edit();
        editor.clear();
        editor.apply();

        // Prepare the worker thread for mode loading and inference
        receiver = new Handler() {
            @Override
            public void handleMessage(Message msg) {
                switch (msg.what) {
                    case RESPONSE_LOAD_MODEL_SUCCESSED:
                        pbLoadModel.dismiss();
                        onLoadModelSuccessed();
                        break;
                    case RESPONSE_LOAD_MODEL_FAILED:
                        pbLoadModel.dismiss();
                        Toast.makeText(MainActivity.this, "Load model failed!", Toast.LENGTH_SHORT).show();
                        onLoadModelFailed();
                        break;
                    case RESPONSE_RUN_MODEL_SUCCESSED:
                        pbRunModel.dismiss();
                        onRunModelSuccessed();
                        break;
                    case RESPONSE_RUN_MODEL_FAILED:
                        pbRunModel.dismiss();
                        Toast.makeText(MainActivity.this, "Run model failed!", Toast.LENGTH_SHORT).show();
                        onRunModelFailed();
                        break;
                    default:
                        break;
                }
            }
        };
        worker = new HandlerThread("Predictor Worker");
        worker.start();
        sender = new Handler(worker.getLooper()) {
            public void handleMessage(Message msg) {
                switch (msg.what) {
                    case REQUEST_LOAD_MODEL:
                        // Load model and reload test image
                        if (onLoadModel()) {
                            receiver.sendEmptyMessage(RESPONSE_LOAD_MODEL_SUCCESSED);
                        } else {
                            receiver.sendEmptyMessage(RESPONSE_LOAD_MODEL_FAILED);
                        }
                        break;
                    case REQUEST_RUN_MODEL:
                        // Run model if model is loaded
                        if (onRunModel()) {
                            receiver.sendEmptyMessage(RESPONSE_RUN_MODEL_SUCCESSED);
                        } else {
                            receiver.sendEmptyMessage(RESPONSE_RUN_MODEL_FAILED);
                        }
                        break;
                    default:
                        break;
                }
            }
        };

        // Setup the UI components
        ivInputImage = findViewById(R.id.iv_input_image);
        tvTop1Result = findViewById(R.id.tv_top1_result);
        tvInferenceTime = findViewById(R.id.tv_inference_time);
        tvSimilarity = findViewById(R.id.similairy);
        tvIndexName = findViewById(R.id.index_name);
        tv_description = findViewById(R.id.description);

        // 启动时隐藏输入label的输入框和确定按钮
        label_name = findViewById(R.id.label_name);
        label_name.setVisibility(View.INVISIBLE);
        label_botton = findViewById(R.id.label_botton);
        label_botton.setVisibility(View.INVISIBLE);
        cancel_botton = findViewById(R.id.cancel_botton);
        cancel_botton.setVisibility(View.INVISIBLE);

        mImage_add_query = (ImageView) findViewById(R.id.add_query);
        mImage_take_query = (ImageView) findViewById(R.id.take_query);
        mImage_add_gallery = (ImageView) findViewById(R.id.add_gallery);
        mImage_take_gallery = (ImageView) findViewById(R.id.take_gallery);
        mImage_save = (ImageView) findViewById(R.id.save);
        mShow_index = (ImageView) findViewById(R.id.show_index);

        File fileindex = new File(getExternalFilesDir(null) + "/" + "index/latest.index");
        if (fileindex.exists()) {
            this.indexPath = "index/latest.index";
        } else {
            this.indexPath = "index/original.index";
        }
        File filelabel = new File(getExternalFilesDir(null) + "/" + "index/latest.txt");
        if (filelabel.exists()) {
            this.labelPath = "index/latest.txt";
        } else {
            this.labelPath = "index/original.txt";
        }
        mImage_add_query.setOnClickListener(v -> {
            if (requestAllPermissions()) {
                label_name.setVisibility(View.INVISIBLE);
                label_botton.setVisibility(View.INVISIBLE);
                cancel_botton.setVisibility(View.INVISIBLE);
                openQueryPhoto();
            } else {
                Toast.makeText(MainActivity.this, "请开启相机和读写文件权限", Toast.LENGTH_SHORT).show();
            }
        });
        mImage_take_query.setOnClickListener(v -> {
            if (requestAllPermissions()) {
                label_name.setVisibility(View.INVISIBLE);
                label_botton.setVisibility(View.INVISIBLE);
                cancel_botton.setVisibility(View.INVISIBLE);
                takeQueryPhoto();
            } else {
                Toast.makeText(MainActivity.this, "请开启相机和读写文件权限", Toast.LENGTH_SHORT).show();
            }
        });
        mImage_add_gallery.setOnClickListener(v -> {
            if (requestAllPermissions()) {
                label_name.setText("");
                label_name.setVisibility(View.INVISIBLE);
                label_botton.setVisibility(View.INVISIBLE);
                cancel_botton.setVisibility(View.INVISIBLE);
                openGalleryPhoto();
            } else {
                Toast.makeText(MainActivity.this, "请开启相机和读写文件权限", Toast.LENGTH_SHORT).show();
            }
        });
        mImage_take_gallery.setOnClickListener(v -> {
            if (requestAllPermissions()) {
                label_name.setVisibility(View.INVISIBLE);
                label_botton.setVisibility(View.INVISIBLE);
                cancel_botton.setVisibility(View.INVISIBLE);
                takeGalleryPhoto();
            } else {
                Toast.makeText(MainActivity.this, "请开启相机和读写文件权限", Toast.LENGTH_SHORT).show();
            }
        });
        mImage_save.setOnClickListener(v -> {
            if (requestAllPermissions()) {
                label_name.setVisibility(View.INVISIBLE);
                label_botton.setVisibility(View.INVISIBLE);
                cancel_botton.setVisibility(View.INVISIBLE);
                saveIndex();
            } else {
                Toast.makeText(MainActivity.this, "请开启相机和读写文件权限", Toast.LENGTH_SHORT).show();
            }
        });

        mShow_index.setOnClickListener(view -> {
//                String fullpath = getExternalFilesDir(null) + "/" + labelPath;
            if (requestAllPermissions()) {
                String labelfile_content = predictor.getClassname();
                AlertDialog alertDialog = new AlertDialog.Builder(MainActivity.this)
                        //标题
                        .setTitle(labelPath)
                        //内容
                        .setMessage(labelfile_content)
                        //图标
                        .setIcon(R.mipmap.ic_launcher)
                        .setPositiveButton("确认", null)
                        .create();
                alertDialog.show();
            } else {
                Toast.makeText(MainActivity.this, "请开启相机和读写文件权限", Toast.LENGTH_SHORT).show();
            }
        });
    }


    @Override
    protected void onResume() {
        super.onResume();
        SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this);
        boolean settingsChanged;
        String model_path = sharedPreferences.getString(getString(R.string.MODEL_PATH_KEY),
                getString(R.string.MODEL_PATH_DEFAULT));
        String label_path = sharedPreferences.getString(getString(R.string.LABEL_PATH_KEY),
                getString(R.string.LABEL_PATH_DEFAULT));
        String index_path = sharedPreferences.getString(getString(R.string.INDEX_PATH_KEY),
                getString(R.string.INDEX_PATH_DEFAULT));
        File fileindex = new File(getExternalFilesDir(null) + "/" + "index/latest.index");
        if (fileindex.exists()) {
            this.indexPath = "index/latest.index";
        } else {
            this.indexPath = "index/original.index";
        }
        File filelabel = new File(getExternalFilesDir(null) + "/" + "index/latest.txt");
        if (filelabel.exists()) {
            this.labelPath = "index/latest.txt";
        } else {
            this.labelPath = "index/original.txt";
        }

        String image_path = "images/demo.jpg";
        settingsChanged = !model_path.equalsIgnoreCase(modelPath);
        settingsChanged |= !image_path.equalsIgnoreCase(imagePath);
        int cpu_thread_num = Integer.parseInt("4");
        settingsChanged |= cpu_thread_num != cpuThreadNum;
        long[] det_input_shape =
                Utils.parseLongsFromString("1,3,640,640", ",");
        long[] rec_input_shape =
                Utils.parseLongsFromString("1,3,224,224", ",");
        String cpu_power_mode = "LITE_POWER_HIGH";
        settingsChanged |= !cpu_power_mode.equalsIgnoreCase(cpuMode);
        int top_k = Integer.parseInt("3");
        settingsChanged |= top_k != topk;
        settingsChanged |= det_input_shape.length != detinputShape.length;
        settingsChanged |= rec_input_shape.length != recinputShape.length;
        if (!settingsChanged) {
            for (int i = 0; i < det_input_shape.length; i++) {
                settingsChanged |= det_input_shape[i] != detinputShape[i];
            }
            for (int i = 0; i < rec_input_shape.length; i++) {
                settingsChanged |= rec_input_shape[i] != recinputShape[i];
            }
        }
        if (settingsChanged || useGpu) {
            modelPath = model_path;
            imagePath = image_path;
            cpuThreadNum = cpu_thread_num;
            detinputShape = det_input_shape;
            recinputShape = rec_input_shape;
            DetModelPath = modelPath;
            RecModelPath = modelPath;
            topk = top_k;
            cpuMode = cpu_power_mode;
            loadModel();
        }
    }

    public void loadModel() {
        pbLoadModel = ProgressDialog.show(this, "", "Loading model...", false, false);
        sender.sendEmptyMessage(REQUEST_LOAD_MODEL);
    }

    public void runModel() {
        pbRunModel = ProgressDialog.show(this, "", "Running model...", false, false);
        sender.sendEmptyMessage(REQUEST_RUN_MODEL);
    }


    public boolean onLoadModel() {
        Context context = getBaseContext();
        ApplicationInfo info = context.getApplicationInfo();
        int debug_mode = info.flags & ApplicationInfo.FLAG_DEBUGGABLE;
        // push model to sdcard
        String realDetModelDir = getExternalFilesDir(null) + "/" + DetModelPath;
        if (debug_mode != 0) {
            Utils.copyDirectoryFromAssets(this, DetModelPath, realDetModelDir);
        }
        String realRecModelDir = getExternalFilesDir(null) + "/" + RecModelPath;
        if (debug_mode != 0) {
            Utils.copyDirectoryFromAssets(this, RecModelPath, realRecModelDir);
        }
        // push label to sdcard
        String realLabelPath = getExternalFilesDir(null) + "/" + labelPath;
        if (debug_mode != 0) {
            Utils.copyFileFromAssets(this, labelPath, realLabelPath);
        }
        String realIndexDir = getExternalFilesDir(null) + "/" + indexPath;
        if (debug_mode != 0) {
            Utils.copyFileFromAssets(this, indexPath, realIndexDir);
        }
        return predictor.init(realDetModelDir, realRecModelDir, realLabelPath, realIndexDir,
                detinputShape, recinputShape, cpuThreadNum, 0, 1, topk, add_gallery, cpuMode);
    }

    public boolean onRunModel() {
        return predictor.isLoaded() && predictor.process();
    }

    public void onLoadModelSuccessed() {
        // Load test image from path and run model
        try {
            if (imagePath.isEmpty()) {
                return;
            }
            Bitmap image;
            // Read test image file from custom path if the first character of mode path is '/', otherwise read test
            // image file from assets
            if (imagePath.charAt(0) != '/') {
                InputStream imageStream = getAssets().open(imagePath);
                image = BitmapFactory.decodeStream(imageStream);
            } else {
                if (!new File(imagePath).exists()) {
                    return;
                }
                image = BitmapFactory.decodeFile(imagePath);
            }
            if (image != null && predictor.isLoaded()) {
                predictor.setInputImage(image);
                runModel();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void onLoadModelFailed() {
    }

    @SuppressLint("SetTextI18n")
    public void onRunModelSuccessed() {
        // Obtain results and update UI
        tvInferenceTime.setText("预测时间：" + predictor.inferenceTime() + " ms");
        Bitmap inputImage = predictor.inputImage();
        if (inputImage != null) {
            ivInputImage.setImageBitmap(inputImage);
        }
        String res = predictor.top1Result();
        if (res.contains(",")) {
            String[] res_split = res.split(",");
            tvTop1Result.setText("类别：" + res_split[0].trim());
            res_split[1] = res_split[1].trim().substring(0, 5);
            tvSimilarity.setText("相似度：" + res_split[1]);
            tvIndexName.setText("检索库名称：" + indexPath.split("/")[1].split("\\.")[0]);
        }
    }

    public void onRunModelFailed() {
    }

    public void onImageChanged(Bitmap image) {
        // Rerun model if users pick test image from gallery or camera
        if (image != null && predictor.isLoaded()) {
            label_name.setVisibility(View.INVISIBLE);
            label_botton.setVisibility(View.INVISIBLE);
            cancel_botton.setVisibility(View.INVISIBLE);
            predictor.setAddGallery(0);
            predictor.setInputImage(image);
            runModel();
            tv_description.setText("【待识别图片】");
        }
    }

    @SuppressLint("SetTextI18n")
    public void onAddGallery(Bitmap image) {
        if (image != null && predictor.isLoaded()) {
            ivInputImage.setImageBitmap(image);
            tv_description.setText("【待加库图片】");
            predictor.setAddGallery(1);
            predictor.setInputImage(image);
            runModel();
            label_name.setVisibility(View.VISIBLE);
            label_name.setHint("image label name");
            label_name.setText("");
            label_botton.setVisibility(View.VISIBLE);
            cancel_botton.setVisibility(View.VISIBLE);
            tvTop1Result.setText("类别：");
            tvSimilarity.setText("相似度：");
            tvInferenceTime.setText("预测时间：" + predictor.inferenceTime() + " ms");
            cancel_botton.setOnClickListener(view -> {
                predictor.setAddGallery(0);
                label_name.setVisibility(View.INVISIBLE);
                label_botton.setVisibility(View.INVISIBLE);
                cancel_botton.setVisibility(View.INVISIBLE);
                tvTop1Result.setText("类别：");
                tvSimilarity.setText("相似度：");
                tvInferenceTime.setText("预测时间：");
            });
            label_botton.setOnClickListener(view -> {
                predictor.setAddGallery(2);
                predictor.setLabelName(label_name.getText().toString());
                runModel();
                label_name.setVisibility(View.INVISIBLE);
                label_botton.setVisibility(View.INVISIBLE);
                cancel_botton.setVisibility(View.INVISIBLE);
                label_name.setText("");
                tvTop1Result.setText("类别：");
                tvSimilarity.setText("相似度：");
                tvInferenceTime.setText("预测时间：" + predictor.inferenceTime() + " ms");
                Toast.makeText(MainActivity.this, "已添加至检索库", Toast.LENGTH_SHORT).show();
            });
        }
    }


    public void onSettingsClicked() {
        startActivity(new Intent(MainActivity.this, SettingsActivity.class));
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.menu_action_options, menu);
        help = findViewById(R.id.help);
        reset = findViewById(R.id.reset);
        return true;
    }


    public boolean onPrepareOptionsMenu(Menu menu) {
        boolean isLoaded = predictor.isLoaded();
        return super.onPrepareOptionsMenu(menu);
    }

    @SuppressLint("NonConstantResourceId")
    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case android.R.id.home:
                finish();
                break;
            case R.id.reset:
                File filelabel = new File(getExternalFilesDir(null) + "/" + labelPath);
                if (filelabel.exists()) {
                    filelabel.delete();
                }
                File fileindex = new File(getExternalFilesDir(null) + "/" + indexPath);
                if (fileindex.exists()) {
                    fileindex.delete();
                }
                labelPath = getString(R.string.LABEL_PATH_DEFAULT);
                indexPath = getString(R.string.INDEX_PATH_DEFAULT);
                predictor.loadIndex("original");
                tvIndexName.setText("检索库名称：" + indexPath.split("/")[1].split("\\.")[0]);
                Toast.makeText(this, "检索库已初始化为original", Toast.LENGTH_SHORT).show();
                break;
            case R.id.help:
                if (requestAllPermissions()) {
                    // Make sure we have SDCard r&w permissions to load model from SDCard
                    String help_content = "【功能说明】\n本APP基于PaddleClas图像分类开发套件中的通用图像识别系统PP-ShiTu开发，支持对拍照/本地上传的图片进行识别。\n【默认检索库说明】\n · 默认内置检索库名为：original，主要包含常见饮料类别共计196种，可通过 [类别查询] 查看已有类别信息。\n" +
                            " · 可根据实际需求，通过拍照/上传本地图像补充检索库，以提高识别准确率或增加可识别类别。\n注意：修改检索库后需进行保存，否则重启APP后将重置为初始库original。\n";
                    SpannableString span = new SpannableString(help_content);
                    int idx = help_content.indexOf("【功能说明】");
                    span.setSpan(new StyleSpan(Typeface.BOLD), idx, idx + "【功能说明】".length(), Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
                    idx = help_content.indexOf("【默认检索库说明】");
                    span.setSpan(new StyleSpan(Typeface.BOLD), idx, idx + "【默认检索库说明】".length(), Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
                    idx = help_content.indexOf("注意：");
                    span.setSpan(new StyleSpan(Typeface.BOLD), idx, idx + "注意：".length(), Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
                    AlertDialog alertDialog = new AlertDialog.Builder(MainActivity.this)
                            //标题
                            .setTitle("使用说明")
                            //内容
                            .setMessage(span)
                            //图标
                            .setIcon(R.mipmap.ic_launcher)
                            .setPositiveButton("确认", null)
                            .create();
                    alertDialog.show();
                }
                break;
            case R.id.settings:
                if (requestAllPermissions()) {
                    // Make sure we have SDCard r&w permissions to load model from SDCard
                    onSettingsClicked();
                }
                break;
        }
        return super.onOptionsItemSelected(item);
    }


    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (grantResults[0] != PackageManager.PERMISSION_GRANTED || grantResults[1] != PackageManager.PERMISSION_GRANTED) {
            Toast.makeText(this, "Permission Denied", Toast.LENGTH_SHORT).show();
        }
    }

    private boolean requestAllPermissions() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED
                || ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.CAMERA}, 0);
            return false;
        }
        return true;
    }

    private void openQueryPhoto() {
        Intent intent = new Intent(Intent.ACTION_PICK, null); // 选择数据
        intent.setDataAndType(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, "image/*");
        startActivityForResult(intent, OPEN_QUERY_PHOTO_REQUEST_CODE);
    }

    @SuppressLint("QueryPermissionsNeeded")
    private void takeQueryPhoto() {
        Intent takePhotoIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(takePhotoIntent, TAKE_QUERY_PHOTO_REQUEST_CODE);
    }

    private void openGalleryPhoto() {
        // 增加现有图片到库中---主逻辑代码
        Intent intent = new Intent(Intent.ACTION_PICK, null); // 选择数据
        intent.setDataAndType(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, "image/*");
        startActivityForResult(intent, OPEN_GALLERY_PHOTO_REQUEST_CODE);
    }

    @SuppressLint("QueryPermissionsNeeded")
    private void takeGalleryPhoto() {
        // 直接拍一张图片到库中---主逻辑代码
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(intent, TAKE_GALLERY_PHOTO_REQUEST_CODE);
    }

    private void clearIndex() {
        // 清空index主逻辑代码
        predictor.clearFeature();
    }

    @SuppressLint("SetTextI18n")
    private void saveIndex() {
        label_name.setText("latest");
        indexPath = "index/latest.index";
        labelPath = "index/latest.txt";
        predictor.saveIndex(label_name.getText().toString());
        tvIndexName.setText("检索库名称：" + indexPath.split("/")[1].split("\\.")[0]);
        Toast.makeText(MainActivity.this, "检索库保存并更新为latest", Toast.LENGTH_SHORT).show();
    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK && data != null) {
            switch (requestCode) {
                case OPEN_QUERY_PHOTO_REQUEST_CODE:
                    try {
                        ContentResolver resolver = getContentResolver();
                        Uri uri = data.getData();
                        Bitmap image = MediaStore.Images.Media.getBitmap(resolver, uri);
                        String[] proj = {MediaStore.Images.Media.DATA};
                        Cursor cursor = managedQuery(uri, proj, null, null, null);
                        cursor.moveToFirst();
                        onImageChanged(image);
                    } catch (IOException e) {
                        Log.e(TAG, e.toString());
                    }
                    break;
                case OPEN_GALLERY_PHOTO_REQUEST_CODE:
                    try {

                        ContentResolver resolver = getContentResolver();
                        Uri uri = data.getData();
                        Bitmap image = MediaStore.Images.Media.getBitmap(resolver, uri);
                        String[] proj = {MediaStore.Images.Media.DATA};
                        Cursor cursor = managedQuery(uri, proj, null, null, null);
                        cursor.moveToFirst();
                        onAddGallery(image);
                    } catch (IOException e) {
                        Log.e(TAG, e.toString());
                    }
                    break;
                case TAKE_GALLERY_PHOTO_REQUEST_CODE:
                    Bundle gextras = data.getExtras();
                    Bitmap gimage = (Bitmap) gextras.get("data");
                    onAddGallery(gimage);
                    break;
                case TAKE_QUERY_PHOTO_REQUEST_CODE:
                    Bundle extras = data.getExtras();
                    Bitmap image = (Bitmap) extras.get("data");
                    onImageChanged(image);
                    break;
                case CLEAR_FEATURE_REQUEST_CODE:
                    clearIndex();
                    break;
                default:
                    break;
            }
        }
    }

    @Override
    protected void onDestroy() {
        if (predictor != null) {
            predictor.release();
        }
        worker.quit();
        super.onDestroy();
    }
}
