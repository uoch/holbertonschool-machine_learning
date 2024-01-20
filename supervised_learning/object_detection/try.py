# import the necessary packages
import numpy as np
#  Felzenszwalb et al.
def non_max_suppression_slow(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# initialize the list of picked indexes
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list, add the index
		# value to the list of picked indexes, then initialize
		# the suppression list (i.e. indexes that will be deleted)
		# using the last index
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		suppress = [last]
		# loop over all indexes in the indexes list
		for pos in range(0, last):
			# grab the current index
			j = idxs[pos]
			# find the largest (x, y) coordinates for the start of
			# the bounding box and the smallest (x, y) coordinates
			# for the end of the bounding box
			xx1 = max(x1[i], x1[j])
			yy1 = max(y1[i], y1[j])
			xx2 = min(x2[i], x2[j])
			yy2 = min(y2[i], y2[j])
			# compute the width and height of the bounding box
			w = max(0, xx2 - xx1 + 1)
			h = max(0, yy2 - yy1 + 1)
			# compute the ratio of overlap between the computed
			# bounding box and the bounding box in the area list
			overlap = float(w * h) / area[j]
			# if there is sufficient overlap, suppress the
			# current bounding box
			if overlap > overlapThresh:
				suppress.append(pos)
		# delete all indexes from the index list that are in the
		# suppression list
		idxs = np.delete(idxs, suppress)
	# return only the bounding boxes that were picked
	return boxes[pick]

def non_max_suppression(boxes, scores, threshold):
    """
    Perform non-max suppression on a set of bounding boxes and corresponding scores.

    :param boxes: a list of bounding boxes in the format [xmin, ymin, xmax, ymax]
    :param scores: a list of corresponding scores
    :param threshold: the IoU (intersection-over-union) threshold for merging bounding boxes
    :return: a list of indices of the boxes to keep after non-max suppression
    """
    # Sort the boxes by score in descending order
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        for j in order:
            # Calculate the IoU between the two boxes
            intersection = max(0, min(boxes[i][2], boxes[j][2]) - max(boxes[i][0], boxes[j][0])) * \
                           max(0, min(boxes[i][3], boxes[j][3]) - max(boxes[i][1], boxes[j][1]))
            union = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]) + \
                    (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1]) - intersection
            iou = intersection / union

            # Remove boxes with IoU greater than the threshold
            if iou > threshold:
                order.remove(j)
    return keep
        for cls in unique_classes:
            # Filter boxes, classes, and scores for the current class
            idx = np.where(box_classes == cls)
            boxes_of_cls = filtered_boxes[idx]
            classes_of_cls = box_classes[idx]
            scores_of_cls = box_scores[idx]

            # Sort the boxes, classes, and scores for the current class by confidence scores
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

                # Intersection coordinates of the current box with the rest of boxes
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