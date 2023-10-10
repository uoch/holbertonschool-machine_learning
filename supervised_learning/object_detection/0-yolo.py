#!/usr/bin/env python3
import tensorflow.keras as K
import numpy as np
import cv2
import glob
import os


class Yolo:
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Constructor method
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line[:-1] for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
