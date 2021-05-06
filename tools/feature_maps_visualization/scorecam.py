# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from resnet import ResNet50 
import paddle.fluid as fluid

import numpy as np 
import cv2 
import utils
import argparse
from PIL import Image, ImageFilter
import os
import matplotlib.cm as mpl_color_map
import copy

def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_file", type=str)
    parser.add_argument("-p", "--pretrained_model", type=str)
    parser.add_argument("--show", type=str2bool, default=False)
    parser.add_argument("--interpolation", type=int, default=1)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    
    return parser.parse_args()

def create_operators(interpolation=1):
    size = 224
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    img_scale = 1.0 / 255.0

    decode_op = utils.DecodeImage()
    resize_op = utils.ResizeImage(resize_short=256, interpolation=interpolation)
    crop_op = utils.CropImage(size=(size, size))
    normalize_op = utils.NormalizeImage(
        scale=img_scale, mean=img_mean, std=img_std)
    totensor_op = utils.ToTensor()

    return [decode_op, resize_op, crop_op, normalize_op, totensor_op]


def preprocess(fname, ops):
    data = open(fname, 'rb').read()
    for op in ops:
        data = op(data)

    return data


def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on iamge
    print(org_im.size)
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image

def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr

def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)

def save_class_activation_images(org_img, activation_map, file_name="test"):
    """
        Saves cam activation map and activation map on the original image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    if not os.path.exists('../results'):
        os.makedirs('../results')
    # Grayscale activation map
    heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map, 'hsv')
    # Save colored heatmap
    path_to_file = os.path.join('../results', file_name+'_Cam_Heatmap.png')
    save_image(heatmap, path_to_file)
    # Save heatmap on iamge
    path_to_file = os.path.join('../results', file_name+'_Cam_On_Image.png')
    save_image(heatmap_on_image, path_to_file)
    # SAve grayscale heatmap
    path_to_file = os.path.join('../results', file_name+'_Cam_Grayscale.png')
    save_image(activation_map, path_to_file)


def main():
    args = parse_args()
    operators = create_operators(args.interpolation)
    # assign the place
    if args.use_gpu:
        gpu_id = fluid.dygraph.parallel.Env().dev_id
        place = fluid.CUDAPlace(gpu_id)
    else:
        place = fluid.CPUPlace()

    pre_weights_dict = fluid.load_program_state(args.pretrained_model)
    with fluid.dygraph.guard(place):
        #net = ResNet50()
        #net = SE_ResNet50_vd()
        #net = InceptionV4()
        net = VGG11()
        data = preprocess(args.image_file, operators)
        data = np.expand_dims(data, axis=0)
        data = fluid.dygraph.to_variable(data)
        dy_weights_dict = net.state_dict()
        pre_weights_dict_new = {}
        for key in dy_weights_dict:
            weights_name = dy_weights_dict[key].name
            pre_weights_dict_new[key] = pre_weights_dict[weights_name]
        net.set_dict(pre_weights_dict_new)
        net.eval()
        out, fm = net(data)
        #target_class = np.argmax(out.numpy())
        target_class = 55
        target = fm[0]
        cam = np.ones(target.shape[1:], dtype=np.float32)
        for i in range(len(target)):
            # Unsqueeze to 4D
            #saliency_map = fluid.layers.unsqueeze(fluid.layers.unsqueeze(target[i, :, :],0),0)
            saliency_map = target[0]
            # Upsampling to input size
            #saliency_map = fluid.layers.interpolate(saliency_map, size=(224, 224), mode='bilinear', align_corners=False)
            saliency_map = cv2.resize(saliency_map.numpy(), (224, 224), interpolation=cv2.INTER_LINEAR)[np.newaxis, np.newaxis, :]
            if saliency_map.max() == saliency_map.min():
                continue
            # Scale between 0-1
            norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
            # Get the target score
            norm_saliency_map = fluid.dygraph.to_variable(norm_saliency_map)
            w = fluid.layers.softmax(net(data*norm_saliency_map)[1],axis=1)[0][target_class]
            cam += w.numpy() * target[i, :, :].numpy()
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((data.shape[2],
                       data.shape[3]), Image.ANTIALIAS))/255.0
        input_image = cv2.imread(args.image_file)
        save_class_activation_images(Image.fromarray(input_image), cam)

        
if __name__ == "__main__":
    main()
