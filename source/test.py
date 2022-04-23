import tqdm
import tensorflow as tf

from source.model.base_model import BaseModel


class Tester:

    def __init__(self, model: BaseModel):
        self.model = model.get_model()
        self.metrics_list = list()

    @staticmethod
    def evaluate(features, logits, ground_truth):
        # TODO implement
        return features / (logits - ground_truth)

    def test(self, test_dataset: tf.data.Dataset):
        test_step_counter = None
        with tqdm.tqdm(total=test_step_counter) as pbar:
            for features, ground_truth in test_dataset:
                logits = self.model(features)
                metrics = self.evaluate(features, logits, ground_truth)
                pbar.update(1)

    def save_results(self):
        pass
