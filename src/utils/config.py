import os

import logging
from logging import Formatter
from logging.handlers import RotatingFileHandler

import json
from pprint import pprint

from utils.filesystem import create_directories


class AttributeDict(dict):
    def __getattr__(self, item):
        if item in self:
            return self[item]
        else:
            raise AttributeError(f"No such attribute {item}")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, item):
        if item in self:
            del self[item]
        else:
            raise AttributeError(f"No such attribute {item}")


def setup_logging(log_dir):
    log_file_format = "[%(levelname)s] - %(asctime)s - %(name)s - : %(message)s in %(pathname)s:%(lineno)d"
    log_console_format = "[%(levelname)s]: %(message)s"

    # Main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(Formatter(log_console_format))

    exp_file_handler = RotatingFileHandler('{}exp_debug.log'.format(log_dir), maxBytes=10 ** 6, backupCount=5)
    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(Formatter(log_file_format))

    exp_errors_file_handler = RotatingFileHandler('{}exp_error.log'.format(log_dir), maxBytes=10 ** 6, backupCount=5)
    exp_errors_file_handler.setLevel(logging.WARNING)
    exp_errors_file_handler.setFormatter(Formatter(log_file_format))

    main_logger.addHandler(console_handler)
    main_logger.addHandler(exp_file_handler)
    main_logger.addHandler(exp_errors_file_handler)


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file: the path of the config file
    :return: config(namespace), config(dictionary)
    """

    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        try:
            config = json.load(config_file)
            # EasyDict allows to access dict values as attributes (works recursively).
            return AttributeDict(config)
        except ValueError:
            print("Invalid config file")
            exit(-1)


def process_config(json_file):
    config = get_config_from_json(json_file)
    print(" THE Configuration of your experiment ..")
    pprint(config)

    # making sure that you have provided the exp_name.
    try:
        print("The experiment name is {}".format(config.experiment_name))
    except AttributeError:
        print("Invalid config file")
        exit(-1)

    # create some important directories to be used for that experiment.
    config.summary_dir = os.path.join("experiments", config.experiment_name, "summaries/")
    config.checkpoint_dir = os.path.join("experiments", config.experiment_name, "checkpoints/")
    config.out_dir = os.path.join("experiments", config.experiment_name, "out/")
    config.log_dir = os.path.join("experiments", config.experiment_name, "logs/")
    create_directories([config.summary_dir, config.checkpoint_dir, config.out_dir, config.log_dir])

    # setup logging in the project
    setup_logging(config.log_dir)

    logging.getLogger().info("Configuration loaded, the pipeline will begin now.")

    return config
