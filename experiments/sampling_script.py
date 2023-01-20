from typing import Optional
from argparse import ArgumentParser
import ruamel.yaml as yaml

from experiments.config import TrainingConfig, SamplingConfig
from experiments.setup_trainer import setup_trainer


# TODO check if the sampling is not starting at the wrong end
# TODO test on the training dataset to check for overfitting
# TODO check if the condition for the time-dependent loss weighting should be dropped
# TODO check if the validation loss works


def sample(config, test_steps: Optional[int] = None):
    trainer = setup_trainer(config)
    trainer.load(config.load_milestone)
    trainer.test(test_steps=test_steps, results_folder=config.experiments_results_folder)


def run_ablation(training_config: TrainingConfig, sampling_config: SamplingConfig, test_steps: Optional[int] = None):
    sampling_configurations = TrainingConfig.generate_sampling_configs(training_config, sampling_config)
    [sample(config, test_steps=test_steps) for config in sampling_configurations]


def main():
    arg_parser = ArgumentParser("Command line interface for running the training and testing of the model")
    arg_parser.add_argument("-t", "--training-config-file", required=True,
                            type=str,
                            help="path to YAML file specifying this script's training config params",
                            metavar="TRAINING_CONFIG_FILE")
    arg_parser.add_argument("-s", "--sampling-config-file", required=True,
                            type=str,
                            help="path to YAML file specifying this script's sampling config params",
                            metavar="SAMPLING_CONFIG_FILE")
    arg_parser.add_argument("-n", "--num-test-steps", required=False,
                            default=None,
                            type=int,
                            help="The number of test steps. If not provided, defaults to the length of the test dataset",
                            metavar="NUM_TEST_STEPS")
    

    command_line_args = arg_parser.parse_args()
    with open(command_line_args.training_config_file, "r") as file:
        dict_training_config_params = yaml.safe_load(file)
    with open(command_line_args.sampling_config_file, "r") as file:
        dict_sampling_config_params = yaml.safe_load(file)

    training_config = TrainingConfig(**dict_training_config_params)
    sampling_config = SamplingConfig(**dict_sampling_config_params)
    run_ablation(training_config, sampling_config, command_line_args.num_test_steps)


if __name__ == "__main__":
    main()