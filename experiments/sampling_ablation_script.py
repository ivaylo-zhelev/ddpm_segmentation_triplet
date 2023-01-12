from argparse import ArgumentParser
import ruamel.yaml as yaml

from experiments.config import TrainingConfig
from experiments.setup import setup_trainer


def ablate(config: TrainingConfig):
    trainer = setup_trainer(config)
    trainer.load(config.load_milestone)
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
    ablate(config)


if __name__ == "__main__":
    main()