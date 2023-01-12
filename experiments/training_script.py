from typing import Tuple, Union, Optional
from pathlib import Path
from argparse import ArgumentParser

from dataclasses import dataclass, fields, _MISSING_TYPE
import ruamel.yaml as yaml

from denoising_diffusion_pytorch import Unet, GaussianDiffusionSegmentationMapping, TrainerSegmentation

# TODO try TripletSemiHardLoss
# TODO check into weight initialization


@dataclass
class TrainingConfig:
    images_folder: Union[str, Path]
    segmentation_folder: Union[str, Path]
    results_folder: Union[str, Path]

    dim: int = 64
    dim_mults: Tuple = (1, 2, 4, 8)

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

    def __post_init__(self):
        # Loop through the fields
        for field in fields(self):
            # If there is a default and the value of the field is none we can assign a value
            if not isinstance(field.default, _MISSING_TYPE) and getattr(self, field.name) is None:
                setattr(self, field.name, field.default)

        self.images_folder = Path(self.images_folder)
        self.segmentation_folder = Path(self.segmentation_folder)
        self.results_folder = Path(self.results_folder)


def train(config):
    model = Unet(
        dim=config.dim,
        dim_mults=config.dim_mults
    ).cuda()

    diffusion = GaussianDiffusionSegmentationMapping(
        model,
        image_size=config.image_size,
        margin=config.margin,
        regularization_margin=config.regularization_margin,
        regularize_to_white_image=config.regularize_to_white_image,
        loss_type=config.loss_type,
        timesteps=config.timesteps,           
        sampling_timesteps=config.sampling_timesteps,
        noising_timesteps=config.noising_timesteps,
        ddim_sampling_eta=config.ddim_sampling_eta,
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
        optimizer=config.optimizer,
        adam_betas=config.adam_betas,
        lr_decay=config.lr_decay,
        weight_decay=config.weight_decay,
        rms_prop_alpha=config.rms_prop_alpha,
        momentum=config.momentum,
        etas=config.etas,
        step_sizes=config.step_sizes,
        train_lr=config.train_lr,
        train_num_steps=config.train_num_steps,    
        gradient_accumulate_every=config.gradient_accumulate_every,
        ema_decay=config.ema_decay,
        amp=True
    )

    if config.load_milestone:
        trainer.load(config.load_milestone)

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