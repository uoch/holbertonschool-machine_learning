#!/usr/bin/env python3
"""this module contains the class Yolo"""
import tensorflow.keras as K
import numpy as np
import os
import cv2


def sigmoid(x):
    """sigmoid function"""
    return 1 / (1 + np.exp(-x))


class Yolo:
    """Yolo class"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Constructor method
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

    def process_outputs(self, outputs, image_size):
        """Process and normalize the output of the YoloV3 model"""
        boxes = []
        box_confidences = []
        box_class_probs = []
        img_h, img_w = image_size
        i = 0
        for output in outputs:
            grid_h, grid_w, nb_box, _ = output.shape
            box_conf = sigmoid(output[:, :, :, 4:5])
            box_prob = sigmoid(output[:, :, :, 5:])
            box_confidences.append(box_conf)
            box_class_probs.append(box_prob)
            # t_x, t_y : x and y coordinates of the center pt of the anchor box
            # t_w, t_h : width and height of the anchor box
            t_x = output[:, :, :, 0]
            t_y = output[:, :, :, 1]
            t_w = output[:, :, :, 2]
            t_h = output[:, :, :, 3]
            # c_x, c_y : represents the grid of model
            c_x = np.arange(grid_w)
            c_x = np.tile(c_x, grid_h)
            c_x = c_x.reshape(grid_h, grid_w, 1)

            c_y = np.arange(grid_h)
            c_y = np.tile(c_y, grid_w)
            c_y = c_y.reshape(1, grid_h, grid_w).T

            # p_w, p_h : anchors dimensions in the c

            p_w = self.anchors[i, :, 0]
            p_h = self.anchors[i, :, 1]

            # yolo formula (get the coordinates in the prediction box)
            b_x = (sigmoid(t_x) + c_x)
            b_y = (sigmoid(t_y) + c_y)
            b_w = (np.exp(t_w) * p_w)
            b_h = (np.exp(t_h) * p_h)
            # normalize to the input size
            b_x = b_x / grid_w
            b_y = b_y / grid_h
            b_w = b_w / self.model.input.shape[1]
            b_h = b_h / self.model.input.shape[2]
            # scale to the image size (in pixels)
            # top left corner
            x1 = (b_x - b_w / 2) * img_w
            y1 = (b_y - b_h / 2) * img_h
            # bottom right corner
            x2 = (b_x + b_w / 2) * img_w
            y2 = (b_y + b_h / 2) * img_h
            # create the current box
            box = np.zeros((grid_h, grid_w, nb_box, 4))
            box[:, :, :, 0] = x1
            box[:, :, :, 1] = y1
            box[:, :, :, 2] = x2
            box[:, :, :, 3] = y2
            boxes.append(box)
            i += 1
        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter boxes based on class confidence score.

        Args:
            boxes: (list of numpy.ndarray) List of numpy.ndarrays with shape
                (grid_height, grid_width, anchor_boxes, 4) containing the
                processed boundary boxes for each output.
            box_confidences: (list of numpy.ndarray) List of np with shape
                            (grid_height, grid_width, anchor_boxes, 1)
            box_class_probs: (list of numpy.ndarray) List of np with shape
                            (grid_height, grid_width, anchor_boxes, classes)
                            the processed box class probabilities for output.

        Returns:
            - filtered_boxes: (?,4) ? = num of boxes, 4 = coordinates
            - box_classes: (?,) ? = num of boxes and contains the class number
            - box_scores: (?,) ? = num of boxes and contains the box scores
        """

        # Extract confidence scores for each class
        class_t = self.class_t
        scores = []
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            # box_conf = conf_prob for box i
            box_conf = box_confidences[i][..., 0]
            # box_class_prob = class_prob for box i
            box_class_prob = box_class_probs[i]
            # box_class_indices = class index with highest score for box i
            class_indices = np.argmax(box_class_prob, axis=-1)
            # class_prob = highest score for box i
            class_prob = np.max(box_class_prob, axis=-1)
            # score for box i
            score = box_conf * class_prob

            # Filter based on the class threshold
            # mask = boolean variable that tells if the score >= class_t
            mask = score >= class_t
            scores.append(score[mask])
            filtered_boxes.append(boxes[i][mask])
            box_classes.append(class_indices[mask])
            box_scores.append(score[mask])

        # Concatenate results
        scores = np.concatenate(scores)
        filtered_boxes = np.concatenate(filtered_boxes)
        box_classes = np.concatenate(box_classes)
        box_scores = np.concatenate(box_scores)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Non-max suppression.
           filtered_boxes: (?, 4) contains all filtered bounding boxes
              box_classes: (?,) contains the class number for the class that
                            filtered_boxes predicts, respectively
                box_scores: (?,) contains the box scores for each box in
                            filtered_boxes, respectively
            returns a tuple of
                (box_predictions, predicted_box_classes, predicted_box_scores)

            """
        nms_t = self.nms_t
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []
        unique_classes = np.unique(box_classes)

        for cls in unique_classes:
            # Filter boxes, classes, and scores for the current class
            idx = np.where(box_classes == cls)
            boxes_of_cls = filtered_boxes[idx]
            classes_of_cls = box_classes[idx]
            scores_of_cls = box_scores[idx]

            # Sort  by confidence scores from high to low
            order = scores_of_cls.argsort()[::-1]
            keep = []

            x1 = boxes_of_cls[:, 0]
            y1 = boxes_of_cls[:, 1]
            x2 = boxes_of_cls[:, 2]
            y2 = boxes_of_cls[:, 3]

            # Calculate areas for all boxes in this class
            areas = (x2 - x1 + 1) * (y2 - y1 + 1)

            while order.shape[0] > 0:
                i = order[0]
                keep.append(i)

                # Intersection coord of the crnt box with the rest of boxes
                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])

                # Intersection width and height
                w = np.maximum(0.0, xx2 - xx1 + 1)
                h = np.maximum(0.0, yy2 - yy1 + 1)

                # Intersection area
                inter = w * h
                all_area = areas[i] + areas[order[1:]] - inter
                overlap = inter / all_area

                # First filter: boxes with overlap > nms_t
                inds = np.where(overlap <= nms_t)[0]
                # Second filter: remove boxes that match the current box
                order = order[inds + 1]

            box_predictions.append(boxes_of_cls[keep])
            predicted_box_classes.append(classes_of_cls[keep])
            predicted_box_scores.append(scores_of_cls[keep])

        box_predictions = np.concatenate(box_predictions)
        predicted_box_classes = np.concatenate(predicted_box_classes)
        predicted_box_scores = np.concatenate(predicted_box_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores

    @staticmethod
    def load_images(folder_path):
        """Load images from a folder"""
        if not os.path.exists(folder_path):
            return None
        images = []
        paths = []
        image_paths = os.listdir(folder_path)
        for image in image_paths:
            img = cv2.imread(os.path.join(folder_path, image))
            if img is not None:
                images.append(img)
                paths.append(os.path.join('./yolo', image))
        return (images, paths)

    def preprocess_images(self, images):
        """Resize and rescale the images before process"""
        input_w = self.model.input.shape[1]
        input_h = self.model.input.shape[2]
        image_shapes = []
        pimages = []
        for image in images:
            image_shapes.append(image.shape[:2])
            pimage = cv2.resize(image, (input_w, input_h),
                                interpolation=cv2.INTER_CUBIC)
            pimage = pimage / 255
            pimages.append(pimage)
        return np.array(pimages), np.array(image_shapes)
