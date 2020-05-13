import tensorflow as tf
# ----------------------------------------
from network import YOLONet
import config
# --------------------------------------------------


class YOLOModel(object):

    def __init__(self, config):
        self.num_bbox_per_cell = config.num_bbox_per_cell
        self.cell_size = config.cell_size
        self.idx1 = (self.cell_size ** 2) * 5
        self.idx2 = (self.cell_size ** 2) * 5 * 2
        self._YOLO_net = YOLONet(config)

    def _forward(self, inputs):
        logits = self._YOLO_net(inputs)
        bbox1 = tf.reshape(logits[:self.idx1], )
        bbox2 = logits[self.idx1: self.idx2]
        category = logits[self.idx2:]
        # ----------------------------------------


