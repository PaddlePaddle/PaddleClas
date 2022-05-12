import argparse
import json
import os

import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--yaml_path', type=str, default='../configs/inference_drink.yaml')
    parser.add_argument(
        '--img_dir',
        type=str,
        default=None,
        help='The dir path for inference images')
    parser.add_argument(
        '--img_path',
        type=str,
        default=None,
        help='The dir path for inference images')
    parser.add_argument(
        '--det_model_path',
        type=str,
        default='./det.nb',
        help="The model path for mainbody  detection")
    parser.add_argument(
        '--rec_model_path',
        type=str,
        default='./rec.nb',
        help="The rec model path")
    parser.add_argument(
        '--rec_label_path',
        type=str,
        default='./label.txt',
        help='The rec model label')
    parser.add_argument(
        '--arch',
        type=str,
        default='PicoDet',
        help='The model structure for detection model')
    parser.add_argument(
        '--fpn-stride',
        type=list,
        default=[8, 16, 32, 64],
        help="The fpn strid for detection model")
    parser.add_argument(
        '--keep_top_k',
        type=int,
        default=100,
        help='The params for nms(postprocess for detection)')
    parser.add_argument(
        '--nms-name',
        type=str,
        default='MultiClassNMS',
        help='The nms name for postprocess of detection model')
    parser.add_argument(
        '--nms_threshold',
        type=float,
        default=0.5,
        help='The nms nms_threshold for detection postprocess')
    parser.add_argument(
        '--nms_top_k',
        type=int,
        default=1000,
        help='The nms_top_k in postprocess of detection model')
    parser.add_argument(
        '--score_threshold',
        type=float,
        default=0.3,
        help='The score_threshold for postprocess of detection')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config_yaml = yaml.safe_load(open(args.yaml_path))
    config_json = {}
    config_json["Global"] = {}
    config_json["Global"][
        "infer_imgs"] = args.img_path if args.img_path else config_yaml[
            "Global"]["infer_imgs"]
    if args.img_dir is not None:
        config_json["Global"]["infer_imgs_dir"] = args.img_dir
        config_json["Global"]["infer_imgs"] = None
    else:
        config_json["Global"][
            "infer_imgs"] = args.img_path if args.img_path else config_yaml[
                "Global"]["infer_imgs"]
    config_json["Global"]["batch_size"] = config_yaml["Global"]["batch_size"]
    config_json["Global"]["cpu_num_threads"] = min(
        config_yaml["Global"]["cpu_num_threads"], 4)
    config_json["Global"]["image_shape"] = config_yaml["Global"]["image_shape"]
    config_json["Global"]["det_model_path"] = args.det_model_path
    config_json["Global"]["rec_model_path"] = args.rec_model_path
    config_json["Global"]["rec_label_path"] = args.rec_label_path
    config_json["Global"]["label_list"] = config_yaml["Global"]["label_list"]
    config_json["Global"]["rec_nms_thresold"] = config_yaml["Global"][
        "rec_nms_thresold"]
    config_json["Global"]["max_det_results"] = config_yaml["Global"][
        "max_det_results"]
    config_json["Global"]["det_fpn_stride"] = args.fpn_stride
    config_json["Global"]["det_arch"] = args.arch
    config_json["Global"]["return_k"] = config_yaml["IndexProcess"]["return_k"]

    # config_json["DetPreProcess"] = config_yaml["DetPreProcess"]
    config_json["DetPreProcess"] = {}
    config_json["DetPreProcess"]["transform_ops"] = []
    for x in config_yaml["DetPreProcess"]["transform_ops"]:
        k = list(x.keys())[0]
        y = x[k]
        y['type'] = k
        config_json["DetPreProcess"]["transform_ops"].append(y)

    config_json["DetPostProcess"] = {
        "keep_top_k": args.keep_top_k,
        "name": args.nms_name,
        "nms_threshold": args.nms_threshold,
        "nms_top_k": args.nms_top_k,
        "score_threshold": args.score_threshold
    }
    #  config_json["RecPreProcess"] = config_yaml["RecPreProcess"]
    config_json["RecPreProcess"] = {}
    config_json["RecPreProcess"]["transform_ops"] = []
    for x in config_yaml["RecPreProcess"]["transform_ops"]:
        k = list(x.keys())[0]
        y = x[k]
        if y is not None:
            y["type"] = k
            config_json["RecPreProcess"]["transform_ops"].append(y)

    # set IndexProces
    config_json["IndexProcess"] = config_yaml["IndexProcess"]
    with open('shitu_config.json', 'w') as fd:
        json.dump(config_json, fd, indent=4)


if __name__ == '__main__':
    main()
