// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <opencv2/core/utils/filesystem.hpp>
#include <ostream>
#include <vector>

#include <cstring>
#include <fstream>
#include <numeric>

#include <auto_log/autolog.h>
#include <gflags/gflags.h>
#include <include/feature_extracter.h>
#include <include/nms.h>
#include <include/object_detector.h>
#include <include/vector_search.h>
#include <include/yaml_config.h>

using namespace std;
using namespace cv;

DEFINE_string(config,
"", "Path of yaml file");
DEFINE_string(c,
"", "Path of yaml file");

void DetPredictImage(const std::vector <cv::Mat> &batch_imgs,
                     const std::vector <std::string> &all_img_paths,
                     const int batch_size, Detection::ObjectDetector *det,
                     std::vector <Detection::ObjectResult> &im_result,
                     std::vector<int> &im_bbox_num, std::vector<double> &det_t,
                     const bool visual_det = false,
                     const bool run_benchmark = false,
                     const std::string &output_dir = "output") {
    int steps = ceil(float(all_img_paths.size()) / batch_size);
    //   printf("total images = %d, batch_size = %d, total steps = %d\n",
    //                 all_img_paths.size(), batch_size, steps);
    for (int idx = 0; idx < steps; idx++) {
        int left_image_cnt = all_img_paths.size() - idx * batch_size;
        if (left_image_cnt > batch_size) {
            left_image_cnt = batch_size;
        }
        // for (int bs = 0; bs < left_image_cnt; bs++) {
        // std::string image_file_path = all_img_paths.at(idx * batch_size+bs);
        // cv::Mat im = cv::imread(image_file_path, 1);
        // batch_imgs.insert(batch_imgs.end(), im);
        // }

        // Store all detected result
        std::vector <Detection::ObjectResult> result;
        std::vector<int> bbox_num;
        std::vector<double> det_times;
        bool is_rbox = false;
        if (run_benchmark) {
            det->Predict(batch_imgs, 10, 10, &result, &bbox_num, &det_times);
        } else {
            det->Predict(batch_imgs, 0, 1, &result, &bbox_num, &det_times);
            // get labels and colormap
            auto labels = det->GetLabelList();
            auto colormap = Detection::GenerateColorMap(labels.size());

            int item_start_idx = 0;
            for (int i = 0; i < left_image_cnt; i++) {
                cv::Mat im = batch_imgs[i];
                int detect_num = 0;

                for (int j = 0; j < bbox_num[i]; j++) {
                    Detection::ObjectResult item = result[item_start_idx + j];
                    if (item.confidence < det->GetThreshold() || item.class_id == -1) {
                        continue;
                    }
                    detect_num += 1;
                    im_result.push_back(item);
                    if (visual_det) {
                        if (item.rect.size() > 6) {
                            is_rbox = true;
                            printf(
                                    "class=%d confidence=%.4f rect=[%d %d %d %d %d %d %d %d]\n",
                                    item.class_id, item.confidence, item.rect[0], item.rect[1],
                                    item.rect[2], item.rect[3], item.rect[4], item.rect[5],
                                    item.rect[6], item.rect[7]);
                        } else {
                            printf("class=%d confidence=%.4f rect=[%d %d %d %d]\n",
                                   item.class_id, item.confidence, item.rect[0], item.rect[1],
                                   item.rect[2], item.rect[3]);
                        }
                    }
                }
                im_bbox_num.push_back(detect_num);
                item_start_idx = item_start_idx + bbox_num[i];

                // Visualization result
                if (visual_det) {
                    std::cout << all_img_paths.at(idx * batch_size + i)
                              << " The number of detected box: " << detect_num
                              << std::endl;
                    cv::Mat vis_img = Detection::VisualizeResult(im, im_result, labels,
                                                                 colormap, is_rbox);
                    std::vector<int> compression_params;
                    compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
                    compression_params.push_back(95);
                    std::string output_path(output_dir);
                    if (output_dir.rfind(OS_PATH_SEP) != output_dir.size() - 1) {
                        output_path += OS_PATH_SEP;
                    }
                    std::string image_file_path = all_img_paths.at(idx * batch_size + i);
                    output_path +=
                            image_file_path.substr(image_file_path.find_last_of('/') + 1);
                    cv::imwrite(output_path, vis_img, compression_params);
                    printf("Visualized output saved as %s\n", output_path.c_str());
                }
            }
        }
        det_t[0] += det_times[0];
        det_t[1] += det_times[1];
        det_t[2] += det_times[2];
    }
}

void PrintResult(std::string &img_path,
                 std::vector <Detection::ObjectResult> &det_result,
                 std::vector<int> &indeices, VectorSearch &vector_search,
                 SearchResult &search_result) {
    printf("%s:\n", img_path.c_str());
    for (int i = 0; i < indeices.size(); ++i) {
        int t = indeices[i];
        printf("\tresult%d: bbox[%d, %d, %d, %d], score: %f, label: %s\n", i,
               det_result[t].rect[0], det_result[t].rect[1], det_result[t].rect[2],
               det_result[t].rect[3], det_result[t].confidence,
               vector_search.GetLabel(search_result.I[search_result.return_k * t])
                       .c_str());
    }
}

int main(int argc, char **argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    std::string yaml_path = "";
    if (FLAGS_config == "" && FLAGS_c == "") {
        std::cerr << "[ERROR] usage: " << std::endl
                  << argv[0] << " -c $yaml_path" << std::endl
                  << "or:" << std::endl
                  << argv[0] << " -config $yaml_path" << std::endl;
        exit(1);
    } else if (FLAGS_config != "") {
        yaml_path = FLAGS_config;
    } else {
        yaml_path = FLAGS_c;
    }

    YamlConfig config(yaml_path);
    config.PrintConfigInfo();

    // initialize detector, rec_Model, vector_search
    Feature::FeatureExtracter feature_extracter(config.config_file);
    Detection::ObjectDetector detector(config.config_file);
    VectorSearch searcher(config.config_file);

    // config
    const int batch_size = config.config_file["Global"]["batch_size"].as<int>();
    bool visual_det = false;
    if (config.config_file["Global"]["visual_det"].IsDefined()) {
        visual_det = config.config_file["Global"]["visual_det"].as<bool>();
    }
    bool benchmark = false;
    if (config.config_file["Global"]["benchmark"].IsDefined()) {
        benchmark = config.config_file["Global"]["benchmark"].as<bool>();
    }
    int max_det_results = 5;
    if (config.config_file["Global"]["max_det_results"].IsDefined()) {
        max_det_results = config.config_file["Global"]["max_det_results"].as<int>();
    }
    float rec_nms_thresold = 0.05;
    if (config.config_file["Global"]["rec_nms_thresold"].IsDefined()) {
        rec_nms_thresold =
                config.config_file["Global"]["rec_nms_thresold"].as<float>();
    }

    // load image_file_path
    std::string path =
            config.config_file["Global"]["infer_imgs"].as<std::string>();
    std::vector <std::string> img_files_list;
    if (cv::utils::fs::isDirectory(path)) {
        std::vector <cv::String> filenames;
        cv::glob(path, filenames);
        for (auto f : filenames) {
            img_files_list.push_back(f);
        }
    } else {
        img_files_list.push_back(path);
    }
    std::cout << "img_file_list length: " << img_files_list.size() << std::endl;
    // for time log
    std::vector<double> cls_times = {0, 0, 0};
    std::vector<double> det_times = {0, 0, 0};
    std::vector<double> search_times = {0, 0, 0};
    int instance_num = 0;
    // for read images
    std::vector <cv::Mat> batch_imgs;
    std::vector <std::string> img_paths;
    // for detection
    std::vector <Detection::ObjectResult> det_result;
    std::vector<int> det_bbox_num;
    // for vector search
    std::vector<float> features;
    std::vector<float> feature;
    // for nms
    std::vector<int> indeices;

    int warmup_iter = img_files_list.size() > 5 ? 5 : img_files_list.size();
    if (benchmark) {
        img_files_list.insert(img_files_list.begin(), img_files_list.begin(),
                              img_files_list.begin() + warmup_iter);
    }

    for (int idx = 0; idx < img_files_list.size(); ++idx) {
        std::string img_path = img_files_list[idx];
        cv::Mat srcimg = cv::imread(img_path, cv::IMREAD_COLOR);
        if (!srcimg.data) {
            std::cerr << "[ERROR] image read failed! image path: " << img_path
                      << "\n";
            exit(-1);
        }
        cv::cvtColor(srcimg, srcimg, cv::COLOR_BGR2RGB);

        batch_imgs.push_back(srcimg);
        img_paths.push_back(img_path);

        // step1: get all detection results
        DetPredictImage(batch_imgs, img_paths, batch_size, &detector, det_result,
                        det_bbox_num, det_times, visual_det, false);

        // select max_det_results bbox
        if (det_result.size() > max_det_results) {
            det_result.resize(max_det_results);
        }
        instance_num += det_result.size();

        // step2: add the whole image for recognition to improve recall
        Detection::ObjectResult result_whole_img = {
                {0, 0, srcimg.cols - 1, srcimg.rows - 1}, 0, 1.0};
        det_result.push_back(result_whole_img);
        det_bbox_num[0] = det_result.size() + 1;

        // step3: extract feature for all boxes in an inmage
        SearchResult search_result;
        for (int j = 0; j < det_result.size(); ++j) {
            int w = det_result[j].rect[2] - det_result[j].rect[0];
            int h = det_result[j].rect[3] - det_result[j].rect[1];
            cv::Rect rect(det_result[j].rect[0], det_result[j].rect[1], w, h);
            cv::Mat crop_img = srcimg(rect);
            feature_extracter.Run(crop_img, feature, cls_times);
            features.insert(features.end(), feature.begin(), feature.end());
        }

        // step4: get search result
        auto search_start = std::chrono::steady_clock::now();
        search_result = searcher.Search(features.data(), det_result.size());
        auto search_end = std::chrono::steady_clock::now();

        // nms for search result
        for (int i = 0; i < det_result.size(); ++i) {
            det_result[i].confidence = search_result.D[search_result.return_k * i];
        }
        NMSBoxes(det_result, searcher.GetThreshold(), rec_nms_thresold, indeices);
        auto nms_end = std::chrono::steady_clock::now();
        std::chrono::duration<float> search_diff = search_end - search_start;
        search_times[1] += double(search_diff.count() * 1000);

        std::chrono::duration<float> nms_diff = nms_end - search_end;
        search_times[2] += double(nms_diff.count() * 1000);

        // print result
        if (not benchmark or (benchmark and idx >= warmup_iter))
            PrintResult(img_path, det_result, indeices, searcher, search_result);

        // for postprocess
        batch_imgs.clear();
        img_paths.clear();
        det_bbox_num.clear();
        det_result.clear();
        feature.clear();
        features.clear();
        indeices.clear();
        if (benchmark and warmup_iter == idx + 1) {
            det_times = {0, 0, 0};
            cls_times = {0, 0, 0};
            search_times = {0, 0, 0};
            instance_num = 0;
        }
    }

    if (benchmark) {
        std::string presion = "fp32";
        if (config.config_file["Global"]["use_fp16"].IsDefined() and
            config.config_file["Global"]["use_fp16"].as<bool>())
            presion = "fp16";
        bool use_gpu = config.config_file["Global"]["use_gpu"].as<bool>();
        bool use_tensorrt = config.config_file["Global"]["use_tensorrt"].as<bool>();
        bool enable_mkldnn =
                config.config_file["Global"]["enable_mkldnn"].as<bool>();
        int cpu_num_threads =
                config.config_file["Global"]["cpu_num_threads"].as<int>();
        int batch_size = config.config_file["Global"]["batch_size"].as<int>();
        std::vector<int> shape =
                config.config_file["Global"]["image_shape"].as < std::vector < int >> ();
        std::string det_shape = std::to_string(shape[0]);
        for (int i = 1; i < shape.size(); ++i)
            det_shape = det_shape + ", " + std::to_string(shape[i]);

        AutoLogger autolog_det("Det", use_gpu, use_tensorrt, enable_mkldnn,
                               cpu_num_threads, batch_size, det_shape, presion,
                               det_times, img_files_list.size() - warmup_iter);
        autolog_det.report();
        AutoLogger autolog_rec("Rec", use_gpu, use_tensorrt, enable_mkldnn,
                               cpu_num_threads, batch_size, "3, 224, 224", presion,
                               cls_times, instance_num);
        autolog_rec.report();
        AutoLogger autolog_search("Search", false, use_tensorrt, enable_mkldnn,
                                  cpu_num_threads, batch_size, "dynamic", presion,
                                  search_times, instance_num);
        autolog_search.report();
    }
    return 0;
}
