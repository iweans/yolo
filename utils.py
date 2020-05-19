import tensorflow as tf
# --------------------------------------------------


def calc_IOU(boxes1, boxes2):
    """
    :param boxes1:
        5-D tensor [BATCH_SIZE, 7, 7, 2, 4] ===> (x_center, y_center, w, h)
        [[50, 100, 20, 30], [60, 90, 40, 50]]
        [[40, 110, 22, 33], [70, 80, 44, 55]]

    :param boxes2:
        5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_center, y_center, w, h)

    :return: 4-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]

    """

    boxes1_vertex = tf.stack([
        boxes1[..., 0] - boxes1[..., 2] / 2.0,    # x_left
        boxes1[..., 1] - boxes1[..., 3] / 2.0,    # y_top
        boxes1[..., 0] + boxes1[..., 2] / 2.0,    # x_right
        boxes1[..., 1] + boxes1[..., 3] / 2.0     # y_bottom
    ], axis=-1)

    boxes2_vertex = tf.stack([
        boxes2[..., 0] - boxes2[..., 2] / 2.0,    # x_left
        boxes2[..., 1] - boxes2[..., 3] / 2.0,    # y_top
        boxes2[..., 0] + boxes2[..., 2] / 2.0,    # x_right
        boxes2[..., 1] + boxes2[..., 3] / 2.0     # y_bottom
    ], axis=-1)

    inter_left_top = tf.maximum(boxes1_vertex[..., :2], boxes2_vertex[..., :2])
    inter_right_bottom = tf.minimum(boxes1_vertex[..., 2:], boxes2_vertex[..., 2:])
    intersection = tf.maximum(0.0, inter_right_bottom - inter_left_top)
    inter_square = intersection[..., 0] * intersection[..., 1]

    square1 = boxes1[..., 2] * boxes1[..., 3]
    square2 = boxes2[..., 2] * boxes2[..., 3]
    union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

    return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)


