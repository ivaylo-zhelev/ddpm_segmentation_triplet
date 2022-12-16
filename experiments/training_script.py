from typing import Tuple, Union
from pathlib import Path
from argparse import ArgumentParser

from dataclasses import dataclass, fields, _MISSING_TYPE
import ruamel.yaml as yaml

from denoising_diffusion_pytorch import Unet, GaussianDiffusionSegmentationMapping, TrainerSegmentation

# TODO try TripletMarginLossWithDistance
# TODO try TripletSemiHardLoss
# TODO try training on little data and check for overfitting
# TODO try gradient clipping
# TODO check the implementation of the time-dependent loss
# TODO try hypertune the optimizer


@dataclass
class TrainingConfig:
    images_folder: Union[str, Path]
    segmentation_folder: Union[str, Path]
    results_folder: Union[str, Path]

    dim: int = 64
    dim_mults: Tuple = (1, 2, 4, 8)

    image_size: int = 320
    margin: float = 1.0
    loss_type: str = "regularized_triplet"
    timesteps: int = 1000           
    sampling_timesteps: int = 100
    is_loss_time_dependent: bool = False

    validate_every: int = 2000
    save_every: int = 2000
    data_split: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    num_samples: int = 25
    train_batch_size: int = 8
    train_lr: float = 8e-5
    train_num_steps: int = 100000    
    gradient_accumulate_every: int = 2
    ema_decay: float = 0.995

    def __post_init__(self):
        # Loop through the fields
        for field in fields(self):
            # If there is a default and the value of the field is none we can assign a value
            if not isinstance(field.default, _MISSING_TYPE) and getattr(self, field.name) is None:
                setattr(self, field.name, field.default)


def train(config):
    model = Unet(
        dim=config.dim,
        dim_mults=config.dim_mults
    ).cuda()

    diffusion = GaussianDiffusionSegmentationMapping(
        model,
        image_size=config.image_size,
        margin=config.margin,
        loss_type=config.loss_type,
        timesteps=config.timesteps,           
        sampling_timesteps=config.sampling_timesteps,
        is_loss_time_dependent=config.is_loss_time_dependent
    ).cuda()

    trainer = TrainerSegmentation(
        diffusion,
        config.images_folder,
        config.segmentation_folder,
        augment_horizontal_flip=False,
        results_folder=config.results_folder,
        validate_every=config.validate_every,
        save_every=config.save_every,
        data_split=config.data_split,
        num_samples=config.num_samples,
        train_batch_size=config.train_batch_size,
        train_lr=config.train_lr,
        train_num_steps=config.train_num_steps,    
        gradient_accumulate_every=config.gradient_accumulate_every,
        ema_decay=config.ema_decay,
        amp=True
    )

    trainer.train()
    trainer.test()


def main():
    arg_parser = ArgumentParser("Command line interface for running the training and testing of the model")
    arg_parser.add_argument("-c", "--config-file", required=True,
                            type=str,
                            help="path to YAML file specifying this script's config params",
                            metavar="CONFIG_FILE")
    command_line_args = arg_parser.parse_args()
    with open(command_line_args.config_file, "r") as file:
        dict_config_params = yaml.safe_load(file)

    config = TrainingConfig(**dict_config_params)
    train(config)


if __name__ == "__main__":
    main()