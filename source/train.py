import configparser
import tqdm
import tensorflow as tf

from source import optimizers
from source import loss
from source.model.base_model import BaseModel


class Trainer:

    _loss_function_dict = {
        'scce': loss.sparse_categorical_crossentropy.scce()
    }

    _optimizer_object_dict = {
        'adam_01': optimizers.adam.setup_01(),
        'sgd_01': optimizers.sgd.setup_01(),
    }

    def __init__(self,
                 config: configparser.ConfigParser,
                 model: BaseModel,
                 train_dataset: tf.data.Dataset,
                 valid_dataset: tf.data.Dataset,
                 ):
        train_config = config['train']
        loss_name = train_config.get('loss_name')
        optimizer_name = train_config.get('optimizer_name')
        self.epochs = train_config.getint('epochs')

        assert loss_name in Trainer._loss_function_dict.keys(), "Loss function name not found"
        self.loss_fn = Trainer._loss_function_dict[loss_name]

        assert optimizer_name in Trainer._optimizer_object_dict.keys(), \
            "Optimizer object name not found."
        self.optimizer = Trainer._optimizer_object_dict[optimizer_name]

        self.model = model.get_model()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        # TODO add accuracy object
        # TODO define parameters on config

    def train(self):
        training_loss = tf.keras.metrics.Mean(name="training_loss")
        validation_loss = tf.keras.metrics.Mean(name="validation_loss")

        train_steps_counter = None
        valid_steps_counter = None

        for epoch in range(self.epochs):
            training_loss.reset_states()
            validation_loss.reset_states()

            with tqdm.tqdm(total=train_steps_counter) as pbar:
                train_steps_counter = 0
                for features, ground_truth in self.train_dataset:
                    with tf.GradientTape as tape:
                        logits = self.model(features)
                        loss = self.loss_fn(logits, ground_truth)
                    training_loss(loss)
                    gradients = tape.gradient(loss, self.model.trainable_variables)

                    train_loss = training_loss.result()
                    train_steps_counter += 1
                    pbar.set_description(
                        f"Training - Epoch: {epoch + 1}, Loss: {train_loss:.3f}"
                        # f"Accuracy: {train_accuracy:.3f}"
                    )

                    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            with tqdm.tqdm(total=valid_steps_counter) as pbar:
                valid_steps_counter = 0
                for features, ground_truth in self.valid_dataset:
                    logits = self.model(features)
                    loss = self.loss_fn(logits, ground_truth)
                    validation_loss(loss)

                    valid_loss = validation_loss.result()
                    valid_steps_counter += 1
                    pbar.set_description(
                        f"Training - Epoch: {epoch + 1}, Loss: {valid_loss:.3f}"
                        # f"Accuracy: {train_accuracy:.3f}"
                    )
        return self.model
