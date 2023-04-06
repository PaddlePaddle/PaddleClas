// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

using System;
using System.IO;
using System.Runtime.InteropServices;
using OpenCvSharp;
using fastdeploy;

namespace Test
{
    public class TestPaddleClas
    {
        public static void Main(string[] args)
        {
            if (args.Length < 3) {
                Console.WriteLine(
                    "Usage: infer_demo path/to/model_dir path/to/image run_option, " +
                    "e.g ./infer_model ./ppyolo_dirname ./test.jpeg 0"
                );
                Console.WriteLine( "The data type of run_option is int, 0: run with cpu; 1: run with gpu");
                return;
            }
            string model_dir = args[0];
            string image_path = args[1];
            string model_file = model_dir + "\\" + "inference.pdmodel";
            string params_file = model_dir + "\\" + "inference.pdiparams";
            string config_file = model_dir + "\\" + "inference_cls.yaml";
            RuntimeOption runtimeoption = new RuntimeOption();
            int device_option = Int32.Parse(args[2]);
            if(device_option==0){
                runtimeoption.UseCpu();
            }else{
                runtimeoption.UseGpu();
            }
            fastdeploy.vision.classification.PaddleClasModel model = new fastdeploy.vision.classification.PaddleClasModel(model_file, params_file, config_file, runtimeoption, ModelFormat.PADDLE);
            if(!model.Initialized()){
                Console.WriteLine("Failed to initialize.\n");
            }
            Mat image = Cv2.ImRead(image_path);
            fastdeploy.vision.ClassifyResult res = model.Predict(image);
            Console.WriteLine(res.ToString());

        }

    }
}