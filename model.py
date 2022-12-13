import tensorflow as tf

class NN(tf.keras.Model):
    def __init__(self):
        self.nn = tf.keras.Sequential([
            tf.keras.layers.Input(2),
            tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(1)])

    def __call__(self, x):
        y = self.nn(x["X"])
        return y