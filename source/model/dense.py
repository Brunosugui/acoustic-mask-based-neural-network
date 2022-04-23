import tensorflow as tf

from source.model.base_model import BaseModel


class DenseModel(BaseModel):

    def __init__(self, input_shape):
        super(DenseModel, self).__init__(input_shape)

    def build(self):
        model_inputs = tf.keras.Layers.Input(self.input_shape)

        dense_1 = tf.keras.Layers.Dense(1024)(model_inputs)

        dense_2 = tf.keras.Layers.Dense(1024)(dense_1)

        dense_3 = tf.keras.Layers.Dense(1024)(dense_2)

        outputs = tf.keras.Layers.Dense(257)(dense_3)

        return tf.keras.Model(model_inputs, outputs)
