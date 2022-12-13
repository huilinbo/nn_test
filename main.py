import tensorflow as tf
import numpy as np

# 加减法 a,b,x

# 乘除法 a,b,x 

# 一元二次方程求解 a,b,c

model = tf.keras.Sequential([
        tf.keras.layers.Input(2),
        tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu),
        tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu),
        tf.keras.layers.Dense(1)])

optimizer = tf.keras.optimizers.Adam()


def get_dataset(N, scale):
    x = scale * np.random.uniform(size=(N, 2))
    # y = np.reshape(5 * np.sqrt(x[:, 0]) + pow(x[:, 1], 3) + x[:, 0] * x[:, 1], (np.shape(x)[0], 1))
    # y = np.reshape( x[:, 0] + np.sqrt(x[:, 1]), (np.shape(x)[0], 1))
    y = np.reshape( x[:, 0] * x[:, 1], (np.shape(x)[0], 1))
    dataset = tf.data.Dataset.from_tensor_slices(
    {
        "X": x,
        "Y": y
    })
    dataset = dataset.repeat()
    dataset = dataset.batch(32)
    return iter(dataset)


def create_loss(pred, true):
    mse = tf.math.reduce_mean(tf.keras.metrics.mean_squared_error(true, pred))
    return mse


def update_step(inputs):
    with tf.GradientTape() as tape:
        prediction = model(inputs["X"])
        loss = create_loss(prediction, inputs["Y"])
    gradients = tape.gradient([loss], model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def validation(sample):
    val_op = open("./validation", "w")
    pred = model(sample["X"])
    for r, p in zip(sample["Y"], pred):
        val_op.write("%.3f %.3f\n"%(r[0], p[0]))
    val_op.flush()


if __name__ == "__main__":
    dataset = get_dataset(1000, 10)
    testset = get_dataset(32, 1000)
    for step in range(100000):
        inputs = next(dataset)
        loss = update_step(inputs)
        if step%100 == 0:
            print("# %4d, loss: %.4f"%(step, loss))
            validation(next(testset))
