import argparse
import os
import pickle

import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    with open(args.config) as fd:
        config = yaml.load(fd.read(), yaml.FullLoader)
    index_dir = ""
    try:
        index_dir = config["IndexProcess"]["index_dir"]
    except Exception as e:
        print("The IndexProcess.index_dir in config_file dose not exist")
        exit(1)
    id_map_path = os.path.join(index_dir, "id_map.pkl")
    assert os.path.exists(
        id_map_path), "The id_map file dose not exist: {}".format(id_map_path)

    with open(id_map_path, "rb") as fd:
        ids = pickle.load(fd)
    with open(os.path.join(index_dir, "id_map.txt"), "w") as fd:
        for k, v in ids.items():
            v = v.split("\t")[1]
            fd.write(str(k) + " " + v + "\n")
    print('Transform id_map sucess')


if __name__ == "__main__":
    main()
