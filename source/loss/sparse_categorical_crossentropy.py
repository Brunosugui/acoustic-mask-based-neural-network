import tensorflow as tf


def scce():
    return tf.keras.losses.SparseCategoricalCrossentropy()
