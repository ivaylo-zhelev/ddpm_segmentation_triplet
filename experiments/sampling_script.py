from argparse import ArgumentParser
import ruamel.yaml as yaml

from experiments.config import TrainingConfig, SamplingConfig
from experiments.setup import setup_trainer


def sample(config):
    trainer.load(config.load_milestone)
    trainer.test(results_folder=config.experiments_results_folder)


def run_ablation(training_config: TrainingConfig, sampling_config: SamplingConfig):
    sampling_configurations = TrainingConfig.generate_sampling_configs(training_config, sampling_config)
    with Pool(sampling_config.num_workers) as p:
        p.map(sample, sampling_configurations)


def main():
    arg_parser = ArgumentParser("Command line interface for running the training and testing of the model")
    arg_parser.add_argument("-t", "--training-config-file", required=True,
                            type=str,
                            help="path to YAML file specifying this script's training config params",
                            metavar="CONFIG_FILE")
    arg_parser.add_argument("-c", "--config-file", required=True,
                            type=str,
                            help="path to YAML file specifying this script's config params",
                            metavar="CONFIG_FILE")

    command_line_args = arg_parser.parse_args()
    with open(command_line_args.training_config_file, "r") as file:
        dict_training_config_params = yaml.safe_load(file)
    with open(command_line_args.sampling_config_file, "r") as file:
        dict_sampling_config_params = yaml.safe_load(file)

    training_config = TrainingConfig(**dict_training_config_params)
    sampling_config = SamplingConfig(**dict_sampling_config_params)
    run_ablation(training_config, sampling_config)


if __name__ == "__main__":
    main()