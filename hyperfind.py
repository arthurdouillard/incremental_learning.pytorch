import argparse
import copy
import os

import ray
import yaml
from ray import tune
from ray.tune import Analysis

import inclearn

INCLEARN_ARGS = vars(inclearn.parser.get_parser().parse_args([]))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-rd", "--ray-directory", default="/data/douillard/ray_results")
    parser.add_argument("-o", "--output-options")
    parser.add_argument("-t", "--tune")
    parser.add_argument("-g", "--gpus", nargs="+", default=["0"])
    parser.add_argument("-per", "--gpu-percent", type=float, default=0.5)
    parser.add_argument("-topn", "--topn", default=5, type=int)
    parser.add_argument("-earlystop", default="ucir", type=str)

    return parser.parse_args()


def train_func(config, reporter):
    train_args = copy.deepcopy(INCLEARN_ARGS)
    train_args.update(config)

    train_args["device"] = [0]
    train_args["threads"] = 2
    train_args["logging"] = "critical"
    train_args["no_progressbar"] = True

    for i, (avg_inc_acc, last_acc, _) in enumerate(inclearn.train.train(train_args)):
        last_acc = last_acc * 100

    reporter(avg_inc_acc=avg_inc_acc)
    return avg_inc_acc


def _get_abs_path(path):
    if path.startswith("/"):
        return path
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), path)


def analyse_ray_dump(ray_directory, topn):
    ea = Analysis(ray_directory)
    trials_dataframe = ea.dataframe()
    trials_dataframe = trials_dataframe.sort_values(by="avg_inc_acc", ascending=False)

    mapping_col_to_index = {}
    result_index = -1
    for index, col in enumerate(trials_dataframe.columns):
        if col.startswith("config:"):
            mapping_col_to_index[col[7:]] = index
        elif col == "avg_inc_acc":
            result_index = index

    print("Best Config:")
    print(
        "avg_inc_acc: {} with {}.".format(
            trials_dataframe.iloc[0][result_index],
            _get_line_results(trials_dataframe, 0, mapping_col_to_index)
        )
    )
    print("\nFollowed by:")
    if topn < 0:
        topn = len(trials_dataframe)
    else:
        topn = min(topn - 1, len(trials_dataframe))

    for i in range(1, topn):
        print(
            "avg_inc_acc: {} with {}.".format(
                trials_dataframe.iloc[i][result_index],
                _get_line_results(trials_dataframe, i, mapping_col_to_index)
            )
        )

    return _get_line_results(trials_dataframe, 0, mapping_col_to_index)


def _get_line_results(df, row_index, mapping):
    results = {}
    for col, index in mapping.items():
        if col.startswith("var:"):
            col = col[4:]

        results[col] = df.iloc[row_index][index]
    return results


def _convert_config(numpy_config):
    config = {}
    for k, v in numpy_config.items():
        if all(not isinstance(v, t) for t in (str, list, dict)):
            v = v.item()
        config[k] = v
    return config


def set_seen_gpus(gpus):
    if len(gpus) >= 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)


def get_tune_config(tune_options):
    with open(tune_options) as f:
        options = yaml.load(f, Loader=yaml.FullLoader)

    config = {}
    for k, v in options.items():
        if not k.startswith("var:"):
            config[k] = v
        else:
            config[k.replace("var:", "")] = tune.grid_search(v)

    return config


def main():
    args = parse_args()

    set_seen_gpus(args.gpus)

    if args.tune is not None:
        config = get_tune_config(args.tune)
        ray.init()
        tune.run(
            train_func,
            name=args.tune.rstrip("/").split("/")[-1],
            stop={"avg_inc_acc": 100},
            config=config,
            resources_per_trial={
                "cpu": 2,
                "gpu": args.gpu_percent
            },
            local_dir=args.ray_directory
        )

        args.ray_directory = os.path.join(args.ray_directory, args.tune.rstrip("/").split("/")[-1])

    if args.ray_directory is not None:
        best_config = analyse_ray_dump(_get_abs_path(args.ray_directory), args.topn)

        if args.output_options:
            with open(args.output_options, "w+") as f:
                yaml.dump(_convert_config(best_config), f)


if __name__ == "__main__":
    main()
