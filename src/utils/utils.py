import numpy as np
from glob import glob
import logging
import sys
import os
import random
import shutil
import yaml
import argparse
import torch
import pickle
from typing import Dict, Any
from distutils.util import strtobool
from PIL import Image


def updated_config() -> Dict[str, Any]:
    # creating an initial parser to read the config.yml file.
    # useful for changing config parameters in bash when running the script
    initial_parser = argparse.ArgumentParser()
    initial_parser.add_argument(
        "--config_path",
        default="src/configs/Ours_ProtoASNet_Video.yml",
        help="Path to a config",
    )
    initial_parser.add_argument(
        "--save_dir",
        default="logs/Video_ProtoASNet/test_run_00",
        help="Path to directory for saving training results",
    )
    initial_parser.add_argument("--eval_only", default=False, help="Evaluate trained model when true")
    initial_parser.add_argument(
        "--eval_data_type",
        default="val",
        help="Data split for evaluation. either val, val_push or test",
    )
    initial_parser.add_argument(
        "--push_only",
        default=False,
        help="Push prototypes if it is true. Useful for pushing a model checkpoint.",
    )
    initial_parser.add_argument(
        "--explain_locally",
        default=False,
        help="Locally explains cases from eval_data_type split",
    )
    initial_parser.add_argument(
        "--explain_globally",
        default=False,
        help="Globally explains the learnt prototypes from the eval_data_type split",
    )
    initial_parser.add_argument(
        "-l",
        "--log_level",
        type=str,
        default="DEBUG",
        help="Logging Level, one of: DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )
    initial_parser.add_argument(
        "-m",
        "--comment",
        type=str,
        default="",
        help="A single line comment for the experiment",
    )

    args, unknown = initial_parser.parse_known_args()

    with open(args.config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    config["config_path"] = args.config_path
    config["save_dir"] = args.save_dir
    config["eval_only"] = args.eval_only
    config["eval_data_type"] = args.eval_data_type
    config["push_only"] = args.push_only
    config["explain_locally"] = args.explain_locally
    config["explain_globally"] = args.explain_globally
    config["log_level"] = args.log_level
    config["comment"] = args.comment

    def get_type_v(v):
        """
        for boolean configs, return a lambda type for argparser so string input can be converted to boolean
        """
        if type(v) == bool:
            return lambda x: bool(strtobool(x))
        else:
            return type(v)

    # creating a final parser with arguments relevant to the config.yml file
    parser = argparse.ArgumentParser()
    for k, v in config.items():
        if type(v) is not dict:
            parser.add_argument(f"--{k}", type=get_type_v(v), default=None)
        else:
            for k2, v2 in v.items():
                if type(v2) is not dict:
                    parser.add_argument(f"--{k}.{k2}", type=get_type_v(v2), default=None)
                else:
                    for k3, v3 in v2.items():
                        if type(v3) is not dict:
                            parser.add_argument(f"--{k}.{k2}.{k3}", type=get_type_v(v3), default=None)
                        else:
                            for k4, v4 in v3.items():
                                parser.add_argument(
                                    f"--{k}.{k2}.{k3}.{k4}",
                                    type=get_type_v(v4),
                                    default=None,
                                )
    args, unknown = parser.parse_known_args()

    # Update the configuration with the python input arguments
    for k, v in config.items():
        if type(v) is not dict:
            if args.__dict__[k] is not None:
                config[k] = args.__dict__[k]
        else:
            for k2, v2 in v.items():
                if type(v2) is not dict:
                    if args.__dict__[f"{k}.{k2}"] is not None:
                        config[k][k2] = args.__dict__[f"{k}.{k2}"]
                else:
                    for k3, v3 in v2.items():
                        if type(v3) is not dict:
                            if args.__dict__[f"{k}.{k2}.{k3}"] is not None:
                                config[k][k2][k3] = args.__dict__[f"{k}.{k2}.{k3}"]
                        else:
                            for k4, v4 in v3.items():
                                if args.__dict__[f"{k}.{k2}.{k3}.{k4}"] is not None:
                                    config[k][k2][k3][k4] = args.__dict__[f"{k}.{k2}.{k3}.{k4}"]
    return config


def set_seed(seed):
    """
    Set up random seed number
    """
    # # Setup random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_save_loc(config):
    save_dir = os.path.join(config["save_dir"])

    #################### Updating the save_dir to avoid overwriting on existing trained models ##################
    # if the save_dir directory exists, find the most recent identifier and increment it
    if os.path.exists(save_dir):
        if os.path.exists(config["model"]["checkpoint_path"]):
            save_dir = os.path.dirname(config["model"]["checkpoint_path"])
            print(
                f"###### Checkpoint '{os.path.basename(config['model']['checkpoint_path'])}'"
                f" provided in path '{save_dir}' ####### \n"
            )
        else:
            print(f"Existing save_dir: {save_dir}\n" f"incrementing the folder number")
            run_id = int(sorted(glob(f"{save_dir[:-3]}*"))[-1][-2:]) + 1
            save_dir = f"{save_dir[:-3]}_{run_id:02}"
            print(f"New location to save the log: {save_dir}")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "img"), exist_ok=True)
    config["save_dir"] = save_dir

    # ############# Document configs ###############
    config_dir = os.path.join(save_dir, "configs")
    os.makedirs(config_dir, exist_ok=True)
    if config["eval_only"]:
        config_path = os.path.join(config_dir, f"eval_{config['eval_data_type']}_config.yml")
    elif config["push_only"]:
        config_path = os.path.join(config_dir, "push_config.yml")
    elif config["explain_locally"]:
        config_path = os.path.join(config_dir, "explain_locally_config.yml")
    elif config["explain_globally"]:
        config_path = os.path.join(config_dir, "explain_globally_config.yml")
    else:
        config_path = os.path.join(config_dir, "train_config.yml")
    with open(config_path, "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


def dict_print(a_dict):
    for k, v in a_dict.items():
        logging.info(f"{k}: {v}")


def print_run_details(config, input_shape):
    print(f"input shape = {input_shape}")


######### Logging
def set_logger(logdir, log_level, filename, comment):
    """
    Set up global logger.
    """
    log_file = os.path.join(logdir, log_level.lower() + f"_{filename}.log")
    logger_format = comment + "| %(asctime)s %(message)s"
    fh = logging.FileHandler(log_file)
    fh.setLevel(log_level)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.DEBUG,
        format=logger_format,
        datefmt="%m-%d %H:%M:%S",
        handlers=[fh, logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger("matplotlib.font_manager").setLevel(logging.INFO)  # remove excessive matplotlib messages
    logging.getLogger("matplotlib").setLevel(logging.INFO)  # remove excessive matplotlib messages
    logging.info("EXPERIMENT BEGIN: " + comment)
    logging.info("logging into %s", log_file)


def backup_code(logdir):
    code_path = os.path.join(logdir, "code")
    dirs_to_save = ["src"]
    os.makedirs(code_path, exist_ok=True)
    # os.system("cp ./*py " + code_path)
    [shutil.copytree(os.path.join("./", this_dir), os.path.join(code_path, this_dir), dirs_exist_ok=True) for this_dir in dirs_to_save]


def print_cuda_statistics():
    import sys
    from subprocess import call
    import torch

    logger = logging.getLogger("Cuda Statistics")
    logger.info("__Python VERSION:  {}".format(sys.version))
    logger.info("__pyTorch VERSION:  {}".format(torch.__version__))
    logger.info("__CUDA VERSION")
    # call(["nvcc", "--version"])
    logger.info("__CUDNN VERSION:  {}".format(torch.backends.cudnn.version()))
    logger.info("__Number CUDA Devices:  {}".format(torch.cuda.device_count()))
    logger.info("__Devices")
    call(
        [
            "nvidia-smi",
            "--format=csv",
            "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free",
        ]
    )
    logger.info("Active CUDA Device: GPU {}".format(torch.cuda.current_device()))
    logger.info("Available devices  {}".format(torch.cuda.device_count()))
    logger.info("Current cuda device  {}".format(torch.cuda.current_device()))


######### ProtoPNet helpers
def makedir(path):
    """
    if path does not exist in the file system, create it
    """
    if not os.path.exists(path):
        os.makedirs(path)


def find_high_activation_crop(activation_map, percentile=95):
    threshold = np.percentile(activation_map, percentile)
    mask = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:, j]) > 0.5:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:, j]) > 0.5:
            upper_x = j
            break
    return lower_y, upper_y + 1, lower_x, upper_x + 1


######## Visualization
def load_image(filepath):
    pil_image = Image.open(filepath)
    return pil_image


def plot_image(ax, pil_image):
    ax.imshow(pil_image)


######## Pickle loading and saving
def load_pickle(pickle_path, log=print):
    with open(pickle_path, "rb") as handle:
        pickle_data = pickle.load(handle)
        log(f"data successfully loaded from {pickle_path}")
    return pickle_data


def save_pickle(pickle_data, pickle_path, log=print):
    with open(pickle_path, "wb") as handle:
        pickle.dump(pickle_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        log(f"data successfully saved in {pickle_path}")
