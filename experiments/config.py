from typing import Tuple, Union, Optional, List
from pathlib import Path
from dataclasses import dataclass, fields, _MISSING_TYPE
from copy import deepcopy
from multiprocessing import Pool

from itertools import product
import numpy as np


@dataclass
class SamplingConfig:
    load_milestone: Union[str, List[int]]

    sampling_timesteps: Union[str, List[int]]
    noising_timesteps: Union[str, List[int]]
    ddim_sampling_eta: Union[str, List[float]]

    num_workers: int = 1

    def __post_init__(self):
        for field in fields(self):
            field_value = getattr(self, field.name)
            if type(field_value) == str:
                try:
                    start, stop, step = field_value.split(":")
                    value_as_list = list(np.arange(float(start), float(stop), float(step)))
                    setattr(self, field.name, value_as_list)
                except AttributeError:
                    assert print(
                        f"{field.name} must be either a list of possible values or follow the format of start:end:step if it is an interval"
                    ) 

        self.num_workers = self.num_workers or 1


@dataclass
class TrainingConfig:
    images_folder: Union[str, Path]
    segmentation_folder: Union[str, Path]
    results_folder: Union[str, Path]

    dim: int = 64
    dim_mults: Tuple[int, int, int, int] = (1, 2, 4, 8)

    image_size: int = 320
    margin: float = 1.0
    regularization_margin: float = 10.0
    regularize_to_white_image: bool = True
    loss_type: str = "regularized_triplet"
    timesteps: int = 1000           
    sampling_timesteps: int = 100
    noising_timesteps: Optional[int] = None
    ddim_sampling_eta: float = 0.0
    is_loss_time_dependent: bool = False

    optimizer: str = "adam"
    adam_betas: Tuple[float, float] = (0.9, 0.99)
    lr_decay: float = 0
    weight_decay: float = 0
    rms_prop_alpha: float = 0.99
    momentum: float = 0
    etas: Tuple[float, float] = (0.5, 1.2)
    step_sizes: Tuple[float, float] = (1e-06, 50)

    validate_every: int = 2000
    save_every: int = 2000
    load_milestone: int = 1
    data_split: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    num_samples: int = 25
    train_batch_size: int = 8
    train_lr: float = 8e-5
    train_num_steps: int = 100000    
    gradient_accumulate_every: int = 2
    ema_decay: float = 0.995

    experiments_results_folder: Optional[Union[str, Path]] = None

    def __post_init__(self):
        # Loop through the fields
        for field in fields(self):
            # If there is a default and the value of the field is none we can assign a value
            if not isinstance(field.default, _MISSING_TYPE) and getattr(self, field.name) is None:
                setattr(self, field.name, field.default)

        self.images_folder = Path(self.images_folder)
        self.segmentation_folder = Path(self.segmentation_folder)
        self.results_folder = Path(self.results_folder)

    def generate_sampling_configs(self, sampling_config):
        sampling_configurations = product(
            sampling_config.sampling_timesteps,
            sampling_config.noising_timesteps,
            sampling_config.ddim_sampling_eta,
            sampling_config.load_milestone    
        )

        training_configs = []
        for sampling_timesteps, noising_timesteps, ddim_sampling_eta, load_milestone in sampling_configurations:
            new_config = deepcopy(self)

            new_config.sampling_timesteps = sampling_timesteps
            new_config.noising_timesteps = noising_timesteps
            new_config.ddim_sampling_eta = ddim_sampling_eta
            new_config.load_milestone = load_milestone

            name_experiment = f"testing_m={load_milestone}_st={sampling_timesteps}_nt={noising_timesteps}_eta={ddim_sampling_eta}"
            new_config.experiments_results_folder = self.results_folder / name_experiment

            training_configs.append(new_config)

        return training_configs
