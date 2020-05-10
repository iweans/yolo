import tensorflow as tf
# --------------------------------------------------


class YOLOBackbone(object):

    def __init__(self, config=None):
        with tf.compat.v1.variable_scope('block01'):
            self._block01_filter = tf.compat.v1.get_variable(shape=(7, 7, 3, 64), dtype=tf.float32,
                                                             initializer=tf.truncated_normal_initializer(0, 0.1),
                                                             name='filter')
        # ----------------------------------------
        with tf.compat.v1.variable_scope('block02'):
            self._block02_filter = tf.compat.v1.get_variable(shape=(3, 3, 64, 192), dtype=tf.float32,
                                                             initializer=tf.truncated_normal_initializer(0, 0.1),
                                                             name='filter')
        # ----------------------------------------
        with tf.compat.v1.variable_scope('block03'):
            self._block03_filter01 = tf.compat.v1.get_variable(shape=(1, 1, 192, 128), dtype=tf.float32,
                                                               initializer=tf.truncated_normal_initializer(0, 0.1),
                                                               name='filter01')
            self._block03_filter02 = tf.compat.v1.get_variable(shape=(3, 3, 128, 256), dtype=tf.float32,
                                                               initializer=tf.truncated_normal_initializer(0, 0.1),
                                                               name='filter02')
            self._block03_filter03 = tf.compat.v1.get_variable(shape=(1, 1, 256, 256), dtype=tf.float32,
                                                               initializer=tf.truncated_normal_initializer(0, 0.1),
                                                               name='filter03')
            self._block03_filter04 = tf.compat.v1.get_variable(shape=(3, 3, 256, 512), dtype=tf.float32,
                                                               initializer=tf.truncated_normal_initializer(0, 0.1),
                                                               name='filter04')
        # ----------------------------------------
        with tf.compat.v1.variable_scope('block04'):
            self._block04_filter01 = tf.compat.v1.get_variable(shape=(1, 1, 512, 256), dtype=tf.float32,
                                                               initializer=tf.truncated_normal_initializer(0, 0.1),
                                                               name='filter01')
            self._block04_filter02 = tf.compat.v1.get_variable(shape=(3, 3, 256, 512), dtype=tf.float32,
                                                               initializer=tf.truncated_normal_initializer(0, 0.1),
                                                               name='filter02')
            self._block04_filter03 = tf.compat.v1.get_variable(shape=(1, 1, 512, 256), dtype=tf.float32,
                                                               initializer=tf.truncated_normal_initializer(0, 0.1),
                                                               name='filter03')
            self._block04_filter04 = tf.compat.v1.get_variable(shape=(3, 3, 256, 512), dtype=tf.float32,
                                                               initializer=tf.truncated_normal_initializer(0, 0.1),
                                                               name='filter04')
            self._block04_filter05 = tf.compat.v1.get_variable(shape=(1, 1, 512, 256), dtype=tf.float32,
                                                               initializer=tf.truncated_normal_initializer(0, 0.1),
                                                               name='filter05')
            self._block04_filter06 = tf.compat.v1.get_variable(shape=(3, 3, 256, 512), dtype=tf.float32,
                                                               initializer=tf.truncated_normal_initializer(0, 0.1),
                                                               name='filter06')
            self._block04_filter07 = tf.compat.v1.get_variable(shape=(1, 1, 512, 256), dtype=tf.float32,
                                                               initializer=tf.truncated_normal_initializer(0, 0.1),
                                                               name='filter07')
            self._block04_filter08 = tf.compat.v1.get_variable(shape=(3, 3, 256, 512), dtype=tf.float32,
                                                               initializer=tf.truncated_normal_initializer(0, 0.1),
                                                               name='filter08')
            self._block04_filter09 = tf.compat.v1.get_variable(shape=(1, 1, 512, 512), dtype=tf.float32,
                                                               initializer=tf.truncated_normal_initializer(0, 0.1),
                                                               name='filter09')
            self._block04_filter10 = tf.compat.v1.get_variable(shape=(3, 3, 512, 1024), dtype=tf.float32,
                                                               initializer=tf.truncated_normal_initializer(0, 0.1),
                                                               name='filter10')
        # ----------------------------------------
        with tf.compat.v1.variable_scope('block05'):
            self._block05_filter01 = tf.compat.v1.get_variable(shape=(1, 1, 1024, 512), dtype=tf.float32,
                                                               initializer=tf.truncated_normal_initializer(0, 0.1),
                                                               name='filter01')
            self._block05_filter02 = tf.compat.v1.get_variable(shape=(3, 3, 512, 1024), dtype=tf.float32,
                                                               initializer=tf.truncated_normal_initializer(0, 0.1),
                                                               name='filter02')
            self._block05_filter03 = tf.compat.v1.get_variable(shape=(1, 1, 1024, 512), dtype=tf.float32,
                                                               initializer=tf.truncated_normal_initializer(0, 0.1),
                                                               name='filter03')
            self._block05_filter04 = tf.compat.v1.get_variable(shape=(3, 3, 512, 1024), dtype=tf.float32,
                                                               initializer=tf.truncated_normal_initializer(0, 0.1),
                                                               name='filter04')
            self._block05_filter05 = tf.compat.v1.get_variable(shape=(3, 3, 1024, 1024), dtype=tf.float32,
                                                               initializer=tf.truncated_normal_initializer(0, 0.1),
                                                               name='filter05')
            self._block05_filter06 = tf.compat.v1.get_variable(shape=(3, 3, 1024, 1024), dtype=tf.float32,
                                                               initializer=tf.truncated_normal_initializer(0, 0.1),
                                                               name='filter06')
        # ----------------------------------------
        with tf.compat.v1.variable_scope('block06'):
            self._block05_filter01 = tf.compat.v1.get_variable(shape=(3, 3, 1024, 1024), dtype=tf.float32,
                                                               initializer=tf.truncated_normal_initializer(0, 0.1),
                                                               name='filter01')
            self._block05_filter02 = tf.compat.v1.get_variable(shape=(3, 3, 512, 1024), dtype=tf.float32,
                                                               initializer=tf.truncated_normal_initializer(0, 0.1),
                                                               name='filter02')


    def __call__(self, inputs):
        # ---------------------------------------- Block01
        with tf.compat.v1.variable_scope('block01'):
            block01_conv = tf.nn.leaky_relu(tf.nn.conv2d(
                    inputs, filter=self._block01_filter, strides=2, padding='SAME', name='conv'),
                name='activ')
            block01_pool = tf.nn.max_pool2d(block01_conv, ksize=2, strides=2, padding='VALID', name='pool')
        feature_map01 = block01_pool
        # ---------------------------------------- Block02
        with tf.compat.v1.variable_scope('block02'):
            block02_conv = tf.nn.leaky_relu(tf.nn.conv2d(
                    block01_pool, filter=self._block02_filter, strides=1, padding='SAME', name='conv'),
                name='activ')
            block02_pool = tf.nn.max_pool2d(block02_conv, ksize=2, strides=2, padding='VALID', name='pool')
        feature_map02 = block02_pool
        # ---------------------------------------- Block03
        with tf.compat.v1.variable_scope('block03'):
            block03_conv01 = tf.nn.leaky_relu(tf.nn.conv2d(
                    block02_pool, filter=self._block03_filter01, strides=1, padding='VALID', name='conv01'),
                name='activ01')
            block03_conv02 = tf.nn.leaky_relu(tf.nn.conv2d(
                    block03_conv01, filter=self._block03_filter02, strides=1, padding='SAME', name='conv02'),
                name='activ02')
            block03_conv03 = tf.nn.leaky_relu(tf.nn.conv2d(
                    block03_conv02, filter=self._block03_filter03, strides=1, padding='VALID', name='conv03'),
                name='activ02')
            block03_conv04 = tf.nn.leaky_relu(tf.nn.conv2d(
                    block03_conv03, filter=self._block03_filter04, strides=1, padding='SAME', name='conv04'),
                name='activ04')
            block03_pool = tf.nn.max_pool2d(block03_conv04, ksize=2, strides=2, padding='VALID', name='pool')
        feature_map03 = block03_pool
        # ---------------------------------------- Block04
        with tf.compat.v1.variable_scope('block04'):
            block04_conv01 = tf.nn.leaky_relu(tf.nn.conv2d(
                    block03_pool, filter=self._block04_filter01, strides=1, padding='VALID', name='conv01'),
                name='activ01')
            block04_conv02 = tf.nn.leaky_relu(tf.nn.conv2d(
                    block04_conv01, filter=self._block04_filter02, strides=1, padding='SAME', name='conv02'),
                name='activ02')
            block04_conv03 = tf.nn.leaky_relu(tf.nn.conv2d(
                    block04_conv02, filter=self._block04_filter03, strides=1, padding='VALID', name='conv03'),
                name='activ03')
            block04_conv04 = tf.nn.leaky_relu(tf.nn.conv2d(
                    block04_conv03, filter=self._block04_filter04, strides=1, padding='SAME', name='conv04'),
                name='activ04')
            block04_conv05 = tf.nn.leaky_relu(tf.nn.conv2d(
                    block04_conv04, filter=self._block04_filter05, strides=1, padding='VALID', name='conv05'),
                name='activ05')
            block04_conv06 = tf.nn.leaky_relu(tf.nn.conv2d(
                    block04_conv05, filter=self._block04_filter06, strides=1, padding='SAME', name='conv06'),
                name='activ06')
            block04_conv07 = tf.nn.leaky_relu(tf.nn.conv2d(
                    block04_conv06, filter=self._block04_filter07, strides=1, padding='VALID', name='conv07'),
                name='activ07')
            block04_conv08 = tf.nn.leaky_relu(tf.nn.conv2d(
                    block04_conv07, filter=self._block04_filter08, strides=1, padding='SAME', name='conv08'),
                name='activ08')
            block04_conv09 = tf.nn.leaky_relu(tf.nn.conv2d(
                    block04_conv08, filter=self._block04_filter09, strides=1, padding='VALID', name='conv09'),
                name='activ09')
            block04_conv10 = tf.nn.leaky_relu(tf.nn.conv2d(
                    block04_conv09, filter=self._block04_filter10, strides=1, padding='SAME', name='conv10'),
                name='activ10')
            block04_pool = tf.nn.max_pool2d(block04_conv10, ksize=2, strides=2, padding='VALID', name='pool')
        feature_map04 = block04_pool
        # ---------------------------------------- Block05
        with tf.compat.v1.variable_scope('block05'):
            block05_conv01 = tf.nn.leaky_relu(tf.nn.conv2d(
                    block04_pool, filter=self._block04_filter01, strides=1, padding='VALID', name='conv01'),
                name='activ01')
            block05_conv02 = tf.nn.leaky_relu(tf.nn.conv2d(
                    block05_conv01, filter=self._block04_filter02, strides=1, padding='SAME', name='conv02'),
                name='activ02')
            block05_conv03 = tf.nn.leaky_relu(tf.nn.conv2d(
                    block05_conv02, filter=self._block04_filter03, strides=1, padding='VALID', name='conv03'),
                name='activ03')
            block05_conv04 = tf.nn.leaky_relu(tf.nn.conv2d(
                    block05_conv03, filter=self._block04_filter04, strides=1, padding='SAME', name='conv04'),
                name='activ04')
            block05_conv05 = tf.nn.leaky_relu(tf.nn.conv2d(
                    block05_conv03, filter=self._block04_filter04, strides=2, padding='SAME', name='conv04'),
                name='activ04')
            block05_conv06 = tf.nn.leaky_relu(tf.nn.conv2d(
                    block05_conv03, filter=self._block04_filter04, strides=2, padding='SAME', name='conv04'),
                name='activ04')
        feature_map05 = block05_conv04
        # ---------------------------------------- Block05
        with tf.compat.v1.variable_scope('block06'):
            block06_conv01 = tf.nn.leaky_relu(tf.nn.conv2d(
                    block05_conv04, filter=self._block04_filter03, strides=1, padding='VALID', name='conv03'),
                name='activ03')

        return block05_conv04



if __name__ == '__main__':
    yolo_backbone = YOLOBackbone()
    inputs = tf.compat.v1.placeholder(tf.float32, (None, 448, 448, 3), name='inputs')
    feature_maps = yolo_backbone(inputs)
    print(feature_maps)
