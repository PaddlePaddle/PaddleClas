from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

import subprocess
import numpy as np

from ppcls.utils import config


def get_result(log_dir):
    log_file = "{}/train.log".format(log_dir)
    with open(log_file, "r") as f:
        raw = f.read()
    res = float(raw.split("best metric: ")[-1].split("]")[0])
    return res


def search_train(search_list,
                 base_program,
                 base_output_dir,
                 search_key,
                 config_replace_value,
                 model_name,
                 search_times=1):
    best_res = 0.
    best = search_list[0]
    all_result = {}
    for search_i in search_list:
        program = base_program.copy()
        for v in config_replace_value:
            program += ["-o", "{}={}".format(v, search_i)]
            if v == "Arch.name":
                model_name = search_i
        res_list = []
        for j in range(search_times):
            output_dir = "{}/{}_{}_{}".format(base_output_dir, search_key,
                                              search_i, j).replace(".", "_")
            program += ["-o", "Global.output_dir={}".format(output_dir)]
            process = subprocess.Popen(program)
            process.communicate()
            res = get_result("{}/{}".format(output_dir, model_name))
            res_list.append(res)
        all_result[str(search_i)] = res_list

        if np.mean(res_list) > best_res:
            best = search_i
            best_res = np.mean(res_list)
    all_result["best"] = best
    return all_result


def search_strategy():
    args = config.parse_args()
    configs = config.get_config(
        args.config, overrides=args.override, show=False)
    base_config_file = configs["base_config_file"]
    distill_config_file = configs.get("distill_config_file", None)
    model_name = config.get_config(base_config_file)["Arch"]["name"]
    gpus = configs["gpus"]
    gpus = ",".join([str(i) for i in gpus])
    base_program = [
        "python3.7", "-m", "paddle.distributed.launch",
        "--gpus={}".format(gpus), "tools/train.py", "-c", base_config_file
    ]
    base_output_dir = configs["output_dir"]
    search_times = configs["search_times"]
    search_dict = configs.get("search_dict")
    all_results = {}
    for search_i in search_dict:
        search_key = search_i["search_key"]
        search_values = search_i["search_values"]
        replace_config = search_i["replace_config"]
        res = search_train(search_values, base_program, base_output_dir,
                           search_key, replace_config, model_name,
                           search_times)
        all_results[search_key] = res
        best = res.get("best")
        for v in replace_config:
            base_program += ["-o", "{}={}".format(v, best)]

    teacher_configs = configs.get("teacher", None)
    if teacher_configs is None:
        print(all_results, base_program)
        return

    algo = teacher_configs.get("algorithm", "skl-ugi")
    supported_list = ["skl-ugi", "udml"]
    assert algo in supported_list, f"algorithm must be in {supported_list} but got {algo}"
    if algo == "skl-ugi":
        teacher_program = base_program.copy()
        # remove incompatible keys
        teacher_rm_keys = teacher_configs["rm_keys"]
        rm_indices = []
        for rm_k in teacher_rm_keys:
            for ind, ki in enumerate(base_program):
                if rm_k in ki:
                    rm_indices.append(ind)
        for rm_index in rm_indices[::-1]:
            teacher_program.pop(rm_index)
            teacher_program.pop(rm_index - 1)
        replace_config = ["Arch.name"]
        teacher_list = teacher_configs["search_values"]
        res = search_train(teacher_list, teacher_program, base_output_dir,
                           "teacher", replace_config, model_name)
        all_results["teacher"] = res
        best = res.get("best")
        t_pretrained = "{}/{}_{}_0/{}/best_model".format(base_output_dir,
                                                         "teacher", best, best)
        base_program += [
            "-o", "Arch.models.0.Teacher.name={}".format(best), "-o",
            "Arch.models.0.Teacher.pretrained={}".format(t_pretrained)
        ]
    elif algo == "udml":
        if "lr_mult_list" in all_results:
            base_program += [
                "-o", "Arch.models.0.Teacher.lr_mult_list={}".format(
                    all_results["lr_mult_list"]["best"])
            ]

    output_dir = "{}/search_res".format(base_output_dir)
    base_program += ["-o", "Global.output_dir={}".format(output_dir)]
    final_replace = configs.get('final_replace')
    for i in range(len(base_program)):
        base_program[i] = base_program[i].replace(base_config_file,
                                                  distill_config_file)
        for k in final_replace:
            v = final_replace[k]
            base_program[i] = base_program[i].replace(k, v)

    process = subprocess.Popen(base_program)
    process.communicate()
    print(all_results, base_program)


if __name__ == '__main__':
    search_strategy()
