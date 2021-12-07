import argparse
from deploy.utils.config import get_config
from tests.infer_speed_test.infer_engines import get_infer_engine
from tests.infer_speed_test.data_loader import get_dataloader


def infer_acc_test():
    pass


def infer_speed_test(config_dict):
    warmup_iters = config_dict["Variables"]["warmup_iters"]
    test_iters = config_dict["Variables"]["test_iters"]
    infer_engine = get_infer_engine(config_dict)
    dataloader = get_dataloader(config_dict)
    infer_engine.mode_warm_up()
    for i in range(warmup_iters):
        input_data = dataloader.gen_data(i)
        infer_engine.infer(input_data)
    infer_engine.mode_speed_test()
    for i in range(test_iters):
        input_data = dataloader.gen_data(i)
        infer_engine.infer(input_data)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument(
        '-o',
        '--override',
        action='append',
        default=[],
        help='config options to be overridden')
    args = parser.parse_args()
    configs = get_config(args.config_path, args.override)
    return configs


if __name__ == '__main__':
    infer_speed_test(parse_args())
