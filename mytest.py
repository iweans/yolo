import tensorflow as tf
tf.enable_eager_execution()
import numpy as np


np.transpose(
    np.reshape(np.array([np.arange(7)] * 7 * 2),    # 14 * 7
               [2, 7, 7]),
    [1, 2, 0]
)

tf.transpose(
    tf.reshape(tf.constant(tf.range))
)

