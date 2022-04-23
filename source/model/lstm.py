import tensorflow as tf

from source.model.base_model import BaseModel


class LSTMModel(BaseModel):

    def __init__(self, input_shape):
        super(LSTMModel, self).__init__(input_shape)

    def build(self):
        model_inputs = tf.keras.Layers.Input(self.input_shape)

        dense_1 = tf.keras.layers.LSTM(256)(model_inputs)

        dense_2 = tf.keras.layers.LSTM(256)(dense_1)

        dense_3 = tf.keras.layers.LSTM(256)(dense_2)

        outputs = tf.keras.layers.LSTM(257)(dense_3)

        return tf.keras.Model(model_inputs, outputs)
