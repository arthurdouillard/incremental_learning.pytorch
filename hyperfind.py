import argparse
import copy
import os

import ray
import yaml
from ray import tune
from ray.tune.analysis import ExperimentAnalysis

import inclearn

INCLEARN_ARGS = vars(inclearn.parser.get_parser().parse_args([]))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-rd", "--ray-directory")
    parser.add_argument("-o", "--output-options")
    parser.add_argument("-t", "--tune")
    parser.add_argument("-g", "--gpus", nargs="+", default=["0"])

    return parser.parse_args()


def train_func(config, reporter):
    train_args = copy.deepcopy(INCLEARN_ARGS)
    train_args.update(config)
    train_args["device"] = 0
    avg_inc_acc = inclearn.train.train(train_args)[0]
    reporter(avg_inc_acc=avg_inc_acc)
    return avg_inc_acc


def _get_abs_path(path):
    if path.startswith("/"):
        return path
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), path)


def analyse_ray_dump(ray_directory):
    ea = ExperimentAnalysis(ray_directory)
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
    for i in range(1, min(4, len(trials_dataframe))):
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
        if all(not isinstance(v, t) for t in (str, list)):
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
            config[k] = tune.grid_search(v)

    return config


def main():
    args = parse_args()

    set_seen_gpus(args.gpus)

    if args.tune is not None:
        ray.init()
        tune.run(
            train_func,
            name=args.tune.rstrip("/").split("/")[-1],
            stop={"avg_inc_acc": 100},
            config=get_tune_config(args.tune),
            resources_per_trial={
                "cpu": 2,
                "gpu": 1
            },
            local_dir="./ray_results"
        )

        args.ray_directory = os.path.join("./ray_results", args.tune.rstrip("/").split("/")[-1])

    if args.ray_directory is not None:
        best_config = analyse_ray_dump(_get_abs_path(args.ray_directory))

        if args.output_options:
            with open(args.output_options, "w+") as f:
                yaml.dump(_convert_config(best_config), f)


if __name__ == "__main__":
    main()
