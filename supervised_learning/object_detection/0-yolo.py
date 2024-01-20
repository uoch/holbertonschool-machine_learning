#!/usr/bin/env python3
"""this module contains the class Yolo"""
import tensorflow.keras as K
import numpy as np
import os


class Yolo:
    """Yolo class"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Constructor method
        model_path: path to where a Darknet Keras model is stored
        classes_path: path to where the list of class names used for the
        Darknet model, listed in order of index, can be found
        class_t: float representing the box score threshold for the initial
        filtering step
        nms_t: float representing the IOU threshold for non-max suppression
        anchors: numpy.ndarray of shape (outputs, anchor_boxes, 2) containing
        all of the anchor boxes:
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError("Wrong model file path")

        if not os.path.exists(classes_path):
            raise FileNotFoundError("Wrong classes file path")
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line[:-1] for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
