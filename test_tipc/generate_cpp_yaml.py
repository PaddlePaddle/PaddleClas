import os
import yaml
import argparse


def str2bool(v):
    if v.lower() == 'true':
        return True
    else:
        return False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', required=True, choices=["cls", "shitu"])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--mkldnn', type=str2bool, default=True)
    parser.add_argument('--gpu', type=str2bool, default=False)
    parser.add_argument('--cpu_thread', type=int, default=1)
    parser.add_argument('--tensorrt', type=str2bool, default=False)
    parser.add_argument('--precision', type=str, choices=["fp32", "fp16"])
    parser.add_argument('--benchmark', type=str2bool, default=True)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument(
        '--cls_yaml_path',
        type=str,
        default="deploy/configs/inference_cls.yaml")
    parser.add_argument(
        '--shitu_yaml_path',
        type=str,
        default="deploy/configs/inference_drink.yaml")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--save_path', type=str, default='./')
    parser.add_argument('--cls_model_dir', type=str)
    parser.add_argument('--det_model_dir', type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.type == "cls":
        save_path = os.path.join(args.save_path,
                                 os.path.basename(args.cls_yaml_path))
        fd = open(args.cls_yaml_path)
    else:
        save_path = os.path.join(args.save_path,
                                 os.path.basename(args.shitu_yaml_path))
        fd = open(args.shitu_yaml_path)
    config = yaml.load(fd, yaml.FullLoader)
    fd.close()

    config["Global"]["batch_size"] = args.batch_size
    config["Global"]["use_gpu"] = args.gpu
    config["Global"]["enable_mkldnn"] = args.mkldnn
    config["Global"]["benchmark"] = args.benchmark
    config["Global"]["use_tensorrt"] = args.tensorrt
    config["Global"]["use_fp16"] = True if args.precision == "fp16" else False
    config["Global"]["gpu_id"] = args.gpu_id
    if args.type == "cls":
        config["Global"]["infer_imgs"] = args.data_dir
        assert args.cls_model_dir
        config["Global"]["inference_model_dir"] = args.cls_model_dir
    else:
        config["Global"]["infer_imgs"] = os.path.join(args.data_dir,
                                                      "test_images")
        config["IndexProcess"]["index_dir"] = os.path.join(args.data_dir,
                                                           "index")
        config["IndexProcess"]["image_root"] = os.path.join(args.data_dir,
                                                            "gallery")
        config["IndexProcess"]["data_file"] = os.path.join(args.data_dir,
                                                           "drink_label.txt")
        assert args.cls_model_dir
        assert args.det_model_dir
        config["Global"]["det_inference_model_dir"] = args.det_model_dir
        config["Global"]["rec_inference_model_dir"] = args.cls_model_dir

    with open(save_path, 'w') as fd:
        yaml.dump(config, fd)
    print("Generate new yaml done")


if __name__ == "__main__":
    main()
