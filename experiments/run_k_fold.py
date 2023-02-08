from typing import Optional
from argparse import ArgumentParser
import ruamel.yaml as yaml

from denoising_diffusion_pytorch import DatasetSegmentation
from experiments.config import TrainingConfig, KFoldConfig
from experiments.setup_trainer import setup_trainer
from experiments.postanalysis_utils import extract_loss_function_cross_fold


def run_k_fold(training_config: TrainingConfig, k_fold_config: KFoldConfig, test_steps: Optional[int] = None):
    k_fold_configurations = TrainingConfig.generate_k_fold_configs(training_config, k_fold_config)
    trainer = None

    for fold_num, config in enumerate(k_fold_configurations):
        print(f"Fold #{fold_num + 1}")
        trainer = setup_trainer(config)
        trainer.train()

    test_ds = DatasetSegmentation(
        images_folder=k_fold_config.testing_images_folder,
        segmentations_folder=k_fold_config.testing_segmentation_folder,
        image_size=training_config.image_size
    )
    trainer.test(test_ds=test_ds, test_steps=test_steps, results_folder=training_config.results_folder)

    extract_loss_function_cross_fold(training_config.results_folder, k=k_fold_config.k)


def main():
    arg_parser = ArgumentParser("Command line interface for running the training and testing of the model")
    arg_parser.add_argument("-t", "--training-config-file", required=True,
                            type=str,
                            help="path to YAML file specifying this script's training config params",
                            metavar="TRAINING_CONFIG_FILE")
    arg_parser.add_argument("-k", "--k_fold-config-file", required=True,
                            type=str,
                            help="path to YAML file specifying this script's K-fold config params",
                            metavar="K_FOLD_CONFIG_FILE")
    arg_parser.add_argument("-n", "--num-test-steps", required=False,
                            default=None,
                            type=int,
                            help="The number of test steps. If not provided, defaults to the length of the test dataset",
                            metavar="NUM_TEST_STEPS")
    

    command_line_args = arg_parser.parse_args()
    with open(command_line_args.training_config_file, "r") as file:
        dict_training_config_params = yaml.safe_load(file)
    with open(command_line_args.k_fold_config_file, "r") as file:
        dict_k_fold_config_params = yaml.safe_load(file)

    training_config = TrainingConfig(**dict_training_config_params)
    k_fold_config = KFoldConfig(**dict_k_fold_config_params)
    run_k_fold(training_config, k_fold_config, command_line_args.num_test_steps)


if __name__ == "__main__":
    main()