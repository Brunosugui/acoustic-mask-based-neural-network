import tensorflow as tf


def setup_01():
    init_learning_rate = 0.005
    decay_steps = 1000000
    decay_rate = 0.96

    learning_rate_scheduler = \
        tf.keras.optimizers.schedules.ExponentialDecay(
            init_learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=True
        )

    optimizer = tf.keras.optimizers.SGD(learning_rate_scheduler)

    return optimizer
