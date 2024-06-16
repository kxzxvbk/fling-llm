import os
import warnings

import torch.multiprocessing as mp
from easydict import EasyDict
from fling.utils import seed_everything
from fling.utils.config_utils import deep_merge_dicts, compile_data_augmentation_config, save_config_file
from zoo.default_config import default_exp_args


def compile_config(new_config: dict, seed: int) -> dict:
    r"""
    Overview:
        This function includes some important steps before the main process starts:
        1) Set the random seed for reproducibility.
        2) Determine the multiprocessing backend.
        3) Merge config (user config & default config).
        4) Compile data augmentation config.
        5) Create logging path and save the compiled config.
    Arguments:
        new_config: user-defined config.
        seed: random seed.
    Returns:
        result_config: the compiled config diction.
    """
    # Set random seed.
    seed_everything(seed)
    # Determine the multiprocessing backend.
    mp.set_start_method('spawn', force=True)

    merged_config = deep_merge_dicts(default_exp_args, new_config)
    result_config = EasyDict(merged_config)

    # Create logging path and save the compiled config.
    exp_dir = result_config.other.logging_path
    if not os.path.exists(exp_dir):
        try:
            os.makedirs(exp_dir)
        except FileExistsError:
            warnings.warn("Logging directory already exists.")
    save_config_file(result_config, os.path.join(exp_dir, 'total_config.py'))

    return result_config
