import subprocess
from ppcls.utils import config


def get_result(log_dir):
    log_file = "{}/train.log".format(log_dir)
    with open(log_file, "r") as f:
        raw = f.read()
    res = float(raw.split("best metric ")[-1].split("]")[0])
    return res


def search_train(search_list, base_program, base_output_dir, search_key, config_replace_value):
    best_res = 0.
    best = search_list[0]
    all_result = {}
    for search_i in search_list:
        program = base_program.copy()
        for v in config_replace_value:
            program += ["-o", "{}={}".format(v, search_i)]
        output_dir = "{}/{}_{}".format(base_output_dir, search_key, search_i.replace(".", "_"))
        program += ["-o", "Global.output_dir={}".format(output_dir)]
        subprocess.Popen(program)
        res = get_result(output_dir)
        all_result[search_i] = res
        if res > best_res:
            best = search_i
            best_res = res
    all_result["best"] = best
    return all_result


def search_strategy():
    args = config.parse_args()
    configs = config.get_config(args.config, overrides=args.override, show=False)
    base_config_file = configs["base_config_file"]
    gpus = configs["gpus"]
    base_program = ["python3.7", "-m", "paddle.distributed.launch", "--gpus={}".format(gpus),
                    "tools/train.py", "-c", base_config_file]
    base_output_dir = configs["output_dir"]
    search_dict = configs.get("search_dict")
    all_results = {}
    for search_key in search_dict:
        search_values = configs[search_key]["search_values"]
        replace_config = search_dict[search_key]["replace_config"]
        res = search_train(search_values, base_program, base_output_dir, search_key, replace_config)
        all_results[search_key] = res
        best = res.get("best")
        for v in replace_config:
            base_program += ["-o", "{}={}".format(v, best)]

    teacher_configs = configs.get("teacher", None)
    if teacher_configs is not None:
        teacher_program = base_program.copy()
        # remove incompatible keys
        teacher_rm_keys = teacher_configs["rm_keys"]
        rm_indices = []
        for rm_k in teacher_rm_keys:
            rm_indices.append(base_program.index(rm_k))
        rm_indices = sorted(rm_indices)
        for rm_index in rm_indices[:, :, -1]:
            teacher_program.pop(rm_index + 1)
            teacher_program.pop(rm_index)
        replace_config = "-o Arch.name"
        teacher_list = teacher_configs["search_values"]
        res = search_train(teacher_list, teacher_program, base_output_dir, "teacher", replace_config)
        all_results["teacher"] = res
        best = res.get("best")
        t_pretrained = "{}/{}_{}".format(base_output_dir, "teacher", best.replace(".", "_"))
        base_program += ["-o", "Arch.models.0.Teacher.name={}".format(best),
                         "-o", "Arch.models.0.Teacher.pretrained={}".format(t_pretrained)]
    output_dir = "{}/search_res".format(base_output_dir)
    base_program += ["-o", "Global.output_dir={}".format(output_dir)]
    subprocess.Popen(base_program)


if __name__ == '__main__':
    search_strategy()
