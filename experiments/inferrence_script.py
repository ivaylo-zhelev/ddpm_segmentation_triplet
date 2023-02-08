from argparse import ArgumentParser
from pathlib import Path

import ruamel.yaml as yaml

from denoising_diffusion_pytorch import DatasetSegmentation
from experiments.config import TrainingConfig
from experiments.setup_trainer import setup_trainer
from experiments.postanalysis_utils import get_last_checkpoint


def infer(config: TrainingConfig):
    original_results_folder = Path(config.results_folder)
    config.results_folder = Path(config.results_folder) / "fold_4"
    trainer = setup_trainer(config)
    trainer.load(get_last_checkpoint(config.results_folder))

    test_ds = DatasetSegmentation(
        images_folder=original_results_folder / "image",
        segmentations_folder=original_results_folder / "ground_truth" ,
        image_size=config.image_size
    )

    trainer.test(test_ds=test_ds, results_folder=original_results_folder)


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
    infer(config)


if __name__ == "__main__":
    main()