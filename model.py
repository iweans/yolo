import tensorflow as tf
# ----------------------------------------
from network import YOLONet
import config
# --------------------------------------------------


class YOLOModel(object):

    def __init__(self, config):
        self.num_bbox_per_cell = config.num_bbox_per_cell
        self.cell_size = config.cell_size
        self.num_classes = config.num_classes
        # ----------------------------------------
        self.idx1 = (self.cell_size ** 2) * self.num_classes
        self.idx2 = self.idx1 + (self.cell_size ** 2) * self.num_bbox_per_cell
        self.x_offset = tf.transpose(tf.reshape(
            tf.repeat(tf.reshape(tf.range(7), (1, 7)),
                      7 * 2, axis=0),
            (2, 7, 7)), (1, 2, 0))
        self.y_offset = tf.transpose(self.x_offset, [1, 0, 2])
        # ----------------------------------------
        self._YOLO_net = YOLONet(config)

    def _forward(self, inputs):
        logits = self._YOLO_net(inputs)
        prob_classes = tf.reshape(self.predicts[0, :self.idx1],
                                  [self.cell_size, self.cell_size, self.num_classes])
        prob_confs = tf.reshape(self.predicts[0, self.idx1:self.idx2],
                                [self.cell_size, self.cell_size, self.num_bbox_per_cell])
        prob_boxes = tf.reshape(self.predicts[0, self.idx2:],
                                [self.cell_size, self.cell_size, self.num_bbox_per_cell, 4])
        # ----------------------------------------
        bnd_boxes = tf.stack([
            (prob_boxes[:, :, :, 0] + tf.constant(self.x_offset, dtype=tf.float32)) / self.cell_size * self.width,
            (prob_boxes[:, :, :, 1] + tf.constant(self.y_offset, dtype=tf.float32)) / self.cell_size * self.height,
            tf.square(prob_boxes[:, :, :, 2]) * self.width,
            tf.square(prob_boxes[:, :, :, 3]) * self.height
        ], axis=3)
        scores = tf.expand_dims(prob_confs, -1) * tf.expand_dims(prob_classes, 2)





    def cal_loss(self):
        pass



if __name__ == '__main__':



