# Based on:
# https://www.pyimagesearch.com/2017/09/18/real-time-object-detection-with-deep-learning-and-opencv/

import numpy as np
import cv2
import os


class Detector(object):

    def __init__(self, use_cuda=True):
        self.model = cv2.dnn.readNetFromCaffe(os.path.join("MobileNet_SSD", "MobileNetSSD_deploy.prototxt.txt"),
                                              os.path.join("MobileNet_SSD", "MobileNetSSD_deploy.caffemodel"))
        if use_cuda:
            try:
                self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            except AttributeError:
                print("Warning: No CUDA compatible OpenCV version found. Using CPU instead of GPU.")
                use_cuda = False
        self.use_cuda = use_cuda
        self.device = "cuda" if use_cuda else "cpu"
        self.class_names = ["background", "aeroplane", "bicycle", "bird", "boat",
                   "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                   "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                   "sofa", "train", "tvmonitor"]

    def __call__(self, img):

        # by resizing to a fixed 300x300 pixels and then normalizing it
        # (note: normalization is done via the authors of the MobileNet SSD implementation)
        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (600, 600)), 0.007843, (600, 600), 127.5)

        # pass the blob through the network and obtain the detections and predictions
        self.model.setInput(blob)
        detections = self.model.forward()
        n_detections = detections.shape[2]
        confidences = np.zeros((n_detections,), dtype=np.float32)
        boxes = np.zeros((n_detections, 4), dtype=np.int)
        labels = np.zeros((n_detections,), dtype=np.int)

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the prediction
            confidences[i] = detections[0, 0, i, 2]

            # extract the index of the class label from the `detections`,
            labels[i] = int(detections[0, 0, i, 1])
            # then compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            boxes[i, ...] = int((startX + endX)/2), int((startY + endY)/2), endX-startX, endY-startY

        return boxes, confidences, labels
