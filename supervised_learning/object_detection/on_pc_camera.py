#!/usr/bin/env python3
#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    import cv2
    # Replace 'your_module' with the actual module name where your Yolo class is defined
    Yolo = __import__('8-yolo').Yolo

    np.random.seed(0)
    anchors = np.array([[[116, 90], [156, 198], [373, 326]],
                        [[30, 61], [62, 45], [59, 119]],
                        [[10, 13], [16, 30], [33, 23]]])

    # Load the YOLO model and configure it
    yolo = Yolo('data/yolo.h5', 'data/coco_classes.txt', 0.6, 0.5, anchors)

    # Set the camera source index (0 for default camera)
    camera_source = 0

    # Open the camera or video source
    source = cv2.VideoCapture(camera_source)

    # Create a window for displaying the camera feed
    win_name = 'Camera Preview'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    while cv2.waitKey(1) != 27:  # Press 'Esc' key to exit
        has_frame, frame = source.read()

        if not has_frame:
            break

        # Perform object detection on the frame
        boxes, box_classes, box_scores = yolo.predict_frame(frame)
        for idx, box in enumerate(boxes):
            top_left_x = int(box[0])
            top_left_y = int(box[1])
            bottom_right_x = int(box[2])
            bottom_right_y = int(box[3])
            class_name = yolo.class_names[box_classes[idx]]
            score = box_scores[idx]
            color = (255, 0, 0)
            cv2.rectangle(frame, (top_left_x, top_left_y),
                          (bottom_right_x, bottom_right_y),
                          color, 2)
            text = class_name + " " + "{:.2f}".format(score)
            cv2.putText(frame, text, (top_left_x, top_left_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                        cv2.LINE_AA)
        cv2.imshow(win_name, frame)

    source.release()
    cv2.destroyAllWindows()
