// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include <ostream>
#include <vector>

#include <cstring>
#include <fstream>
#include <numeric>

#include <include/cls.h>
#include <include/config.h>

using namespace std;
using namespace cv;
using namespace PaddleClas;

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "[ERROR] usage: " << argv[0]
              << " configure_filepath image_path\n";
    exit(1);
  }

  Config config(argv[1]);

  config.PrintConfigInfo();

  std::string img_path(argv[2]);

  cv::Mat srcimg = cv::imread(img_path, cv::IMREAD_COLOR);
  cv::cvtColor(srcimg, srcimg, cv::COLOR_BGR2RGB);

  Classifier classifier(config.cls_model_dir, config.use_gpu, config.gpu_id,
                        config.gpu_mem, config.cpu_math_library_num_threads,
                        config.use_mkldnn, config.use_zero_copy_run,
                        config.resize_short_size, config.crop_size);

  auto start = std::chrono::system_clock::now();
  classifier.Run(srcimg);
  auto end = std::chrono::system_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "Cost "
            << double(duration.count()) *
                   std::chrono::microseconds::period::num /
                   std::chrono::microseconds::period::den
            << " s" << std::endl;

  return 0;
}
