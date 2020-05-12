from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, MaxPooling2D, Dense, Flatten, Reshape
# --------------------------------------------------


class YOLONet(object):

    def __init__(self, config=None):

        self._build_network()

    def _build_network(self, cell_size=7, num_bbox_per_cell=2, num_classes=20, leaky_alpha=0.1):
        num_feature_channel = num_bbox_per_cell*5 + num_classes

        self._network = Sequential([
            Input((448, 448, 3), name='inputs'),
            # ---------------------------------------- block01
            Conv2D(64, 7, strides=2, padding='same', name='block01/convolution'), LeakyReLU(leaky_alpha, name='block01/leaky_relu'),
            MaxPooling2D(2, strides=2, padding='valid', name='block01/max_pooling'),
            # ---------------------------------------- block02
            Conv2D(192, 3, strides=1, padding='same', name='block02/convolution'), LeakyReLU(leaky_alpha, name='block02/leaky_relu'),
            MaxPooling2D(2, strides=2, padding='valid', name='block02/max_pooling'),
            # ---------------------------------------- block03
            Conv2D(128, 1, strides=1, padding='valid', name='block03/convolution01'), LeakyReLU(leaky_alpha, name='block03/leaky_relu01'),
            Conv2D(256, 3, strides=1, padding='same', name='block03/convolution02'), LeakyReLU(leaky_alpha, name='block03/leaky_relu02'),
            Conv2D(256, 1, strides=1, padding='valid', name='block03/convolution03'), LeakyReLU(leaky_alpha, name='block03/leaky_relu03'),
            Conv2D(512, 3, strides=1, padding='same', name='block03/convolution04'), LeakyReLU(leaky_alpha, name='block03/leaky_relu04'),
            MaxPooling2D(2, strides=2, padding='valid', name='block03/max_pooling'),
            # ---------------------------------------- block04
            Conv2D(256, 1, strides=1, padding='valid', name='block04/convolution01'), LeakyReLU(leaky_alpha, name='block04/leaky_relu01'),
            Conv2D(512, 3, strides=1, padding='same', name='block04/convolution02'), LeakyReLU(leaky_alpha, name='block04/leaky_relu02'),
            Conv2D(256, 1, strides=1, padding='valid', name='block04/convolution03'), LeakyReLU(leaky_alpha, name='block04/leaky_relu03'),
            Conv2D(512, 3, strides=1, padding='same', name='block04/convolution04'), LeakyReLU(leaky_alpha, name='block04/leaky_relu04'),
            Conv2D(256, 1, strides=1, padding='valid', name='block04/convolution05'), LeakyReLU(leaky_alpha, name='block04/leaky_relu05'),
            Conv2D(512, 3, strides=1, padding='same', name='block04/convolution06'), LeakyReLU(leaky_alpha, name='block04/leaky_relu06'),
            Conv2D(256, 1, strides=1, padding='valid', name='block04/convolution07'), LeakyReLU(leaky_alpha, name='block04/leaky_relu07'),
            Conv2D(512, 3, strides=1, padding='same', name='block04/convolution08'), LeakyReLU(leaky_alpha, name='block04/leaky_relu08'),
            Conv2D(512, 1, strides=1, padding='valid', name='block04/convolution09'), LeakyReLU(leaky_alpha, name='block04/leaky_relu09'),
            Conv2D(1024, 3, strides=1, padding='same', name='block04/convolution10'), LeakyReLU(leaky_alpha, name='block04/leaky_relu10'),
            MaxPooling2D(2, strides=2, padding='valid', name='block04/max_pooling'),
            # ---------------------------------------- block05
            Conv2D(512, 1, strides=1, padding='valid', name='block05/convolution01'), LeakyReLU(leaky_alpha, name='block05/leaky_relu01'),
            Conv2D(1024, 3, strides=1, padding='same', name='block05/convolution02'), LeakyReLU(leaky_alpha, name='block05/leaky_relu02'),
            Conv2D(512, 1, strides=1, padding='valid', name='block05/convolution03'), LeakyReLU(leaky_alpha, name='block05/leaky_relu03'),
            Conv2D(1024, 3, strides=1, padding='same', name='block05/convolution04'), LeakyReLU(leaky_alpha, name='block05/leaky_relu04'),
            Conv2D(1024, 3, strides=1, padding='same', name='block05/convolution05'), LeakyReLU(leaky_alpha, name='block05/leaky_relu05'),
            Conv2D(1024, 3, strides=2, padding='same', name='block05/convolution06'), LeakyReLU(leaky_alpha, name='block05/leaky_relu06'),
            # ---------------------------------------- block06
            Conv2D(1024, 3, strides=1, padding='same', name='block06/convolution01'), LeakyReLU(leaky_alpha, name='block06/leaky_relu01'),
            Conv2D(1024, 3, strides=1, padding='same', name='block06/convolution02'), LeakyReLU(leaky_alpha, name='block06/leaky_relu02'),
            # ---------------------------------------- block07
            Flatten(name='block07/flatten'),
            Dense(4096, name='block07/fc01'), LeakyReLU(leaky_alpha, name='block07/leaky_relu01'),
            Dense(cell_size*cell_size*num_feature_channel, name='block07/fc02'), LeakyReLU(leaky_alpha, name='block07/leaky_relu02'),
            Reshape((cell_size, cell_size, num_feature_channel), name='block07/reshape')
        ], name='YOLONet')

    def __call__(self, inputs):
        return self._network(inputs)

    def summary(self):
        self._network.summary()


if __name__ == '__main__':
    import tensorflow as tf
    YOLO_net = YOLONet()
    YOLO_net.summary()
    image_batch = tf.placeholder(shape=(4, 448, 448, 3), dtype=tf.float32)
    logits = YOLO_net(image_batch)
    print(logits)
