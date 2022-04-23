import os
import glob
import librosa
import scipy.io
import tensorflow as tf


class FeaturesDataset:

    def __init__(self, config):
        config_data = config['data']
        self.database_path = config_data.get('base_path')

        self.train_base_path = os.path.join(self.database_path, "train")
        self.train_files = glob.glob(os.path.join(self.train_base_path, "*.mat"))

        self.test_base_path = os.path.join(self.database_path, "test")
        self.test_files = glob.glob(os.path.join(self.train_base_path, "*.mat"))

        self.train_data, self.valid_data, self.test_data = None, None, None

        self.prepare_data()

    def features_shape(self):
        pass

    @staticmethod
    def load_mat(path):
        mat = scipy.io.loadmat(path)
        return mat['Mfeat']

    def prepare_data(self):
        # TODO prepare tensor slices for using on tensorflow dataset.
        pass

    @staticmethod
    def create_tensorflow_dataset(tensor_slices):
        dataset = tf.data.Dataset.from_tensor_slices(tensor_slices)
        # TODO map process
        return dataset

    def get_datasets(self):
        train_dataset = self.create_tensorflow_dataset(self.train_data)
        valid_dataset = self.create_tensorflow_dataset(self.valid_data)
        test_dataset = self.create_tensorflow_dataset(self.test_data)

        return train_dataset, valid_dataset, test_dataset
