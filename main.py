import os
import sys
import argparse
import configparser

from source.data.features_dataset import FeaturesDataset
from source.train import Trainer
from source.test import Tester

from source.model import dense
from source.model import lstm

os.environ["TF_DETERMINISTIC_OPS"] = "1"


def setup_tf_options():
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    import tensorflow as tf
    import logging
    logger = tf.get_logger()
    logger.setLevel(logging.ERROR)


def fix_random_seed():
    import os
    import random
    import tensorflow
    import numpy as np

    os.environ["PYTHONHASHSEED"] = "42"
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    tensorflow.random.set_seed(random_seed)


setup_tf_options()
fix_random_seed()

models_dictionary = {
    'dense': dense.DenseModel,
    'lstm': lstm.LSTMModel
}

datasets_dictionary = {
    'abstract': FeaturesDataset,
}


def get_args():
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument('-c', '--config', type=str, default='inputs/configs/default.conf',
                        help="Path to a config file.")
    return vars(parser.parse_args(args=sys.argv[1:]))


def get_config(config_filepath):
    config_parser = configparser.ConfigParser()
    config_parser.read(config_filepath)
    return config_parser


if __name__ == "__main__":
    args = get_args()
    config_path = args.get('config')

    config = get_config(config_path)
    model_name = config['model'].get('model_name')
    dataset_name = config['data'].get('dataset_name')

    dataset = datasets_dictionary[dataset_name](config)

    model_input_shape = dataset.features_shape
    model = models_dictionary[model_name](model_input_shape)

    train_dataset, valid_dataset, test_dataset = dataset.get_datasets()

    trainer = Trainer(config, model, train_dataset, valid_dataset)
    trainer.train()

    tester = Tester(model)
    tester.test(test_dataset)


    print(f"Finished!")
