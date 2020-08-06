# Based on:
# https://github.com/openvinotoolkit/open_model_zoo/tree/master/demos/python_demos/object_detection_demo_ssd_async
# Using models from:
# https://github.com/openvinotoolkit/open_model_zoo/tree/master/demos/pedestrian_tracker_demo

import numpy as np
import cv2
import os
import time


class Detector(object):

    def __init__(self, IECore, conf_thresh, nms_thresh, use_movidius=False):
        self.ie = IECore
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        model_xml = "Openvino/person-detection-retail-0013.xml"
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        self.model = self.ie.read_network(model=model_xml, weights=model_bin)

        self.device = "CPU" if not use_movidius else "MYRIAD"
        self.class_names = ["person"]

        # Verify expected input shape
        for input_key in self.model.input_info:
            if len(self.model.input_info[input_key].input_data.layout) == 4:
                n, self.c, self.h, self.w = self.model.input_info[input_key].input_data.shape

        # Verify the input data expected name
        for input_key in self.model.input_info:
            if len(self.model.input_info[input_key].layout) == 4:
                self.input_name = input_key
                self.model.input_info[input_key].precision = 'U8'

        # Verify the name of the network output blobs
        self.out_blob = next(iter(self.model.outputs))

        # Make model executable
        self.model = self.ie.load_network(network=self.model, device_name=self.device)

    def __call__(self, img):
        # start = time.time()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # The detector expects a BGR image, but the function input is RGB

        # Resize image if original size is different from network input size
        ih, iw = img.shape[:-1]
        if (ih, iw) != (self.h, self.w):
            image = cv2.resize(img, (self.w, self.h))
        image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        image = image[None, ...]  # Add dimension for batch size

        # Convert image to input data
        data = {self.input_name: image}

        # Pass the blob through the network and obtain the detections and predictions
        detections = self.model.infer(inputs=data)[self.out_blob][0][0]
        n_detections = detections.shape[0]
        boxes = np.zeros((n_detections, 6), dtype=np.float)

        for i, proposal in enumerate(detections):
            if proposal[2] > 0:
                startX = np.int(iw * proposal[3])
                startY = np.int(ih * proposal[4])
                endX = np.int(iw * proposal[5])
                endY = np.int(ih * proposal[6])
                # Return XY coordinates of top-left and bottom-right corners, confidence level, and class label
                boxes[i, ...] = startX, startY, endX, endY, proposal[2], 0

        # Eliminate potentially duplicate detections when IoU is over nms_thresh
        boxes = self.iou_filter(boxes, nms_thresh=self.nms_thresh)
        # print("Prediction time: {}s\n".format(time.time() - start))
        if boxes is not None:
            return boxes[:, :4].astype(np.int), boxes[:, 4], boxes[:, 5].astype(np.int)
        return None, None, None

    def iou_filter(self, boxes, nms_thresh):
        sortIds = np.argsort(boxes[:, 4])
        out_boxes = []
        for i in range(boxes.shape[0]):
            box_i = boxes[sortIds[i]]
            if box_i[4] > self.conf_thresh:
                out_boxes.append([
                    int((box_i[0] + box_i[2]) / 2),  # X center
                    int((box_i[1] + box_i[3]) / 2),  # Y center
                    box_i[2] - box_i[0],  # Width
                    box_i[3] - box_i[1],  # Height
                    box_i[4],  # Confidence level
                    box_i[5]  # Class label
                ])
                for j in range(i + 1, boxes.shape[0]):
                    box_j = boxes[sortIds[j]]
                    if self.iou(box_i[:4], box_j[:4]) > nms_thresh:
                        # Put confidence score below 0 so it is ignored
                        boxes[sortIds[j]][4] = -1
        if len(out_boxes) > 0:
            return np.array(out_boxes)
        return None

    def iou(self, box1, box2):
        startX_1, startY_1, endX_1, endY_1 = box1
        startX_2, startY_2, endX_2, endY_2 = box2
        x1 = np.maximum(startX_1, startX_2)
        y1 = np.maximum(startY_1, startY_2)
        x2 = np.minimum(endX_1, endX_2)
        y2 = np.minimum(endY_1, endY_2)
        interArea = np.maximum((x2 - x1 + 1), 0) * np.maximum((y2 - y1 + 1), 0)
        box1Area = (endX_1 - startX_1 + 1) * (endY_1 - startY_1 + 1)
        box2Area = (endX_2 - startX_2 + 1) * (endY_2 - startY_2 + 1)
        iou = interArea / (box1Area + box2Area - interArea)
        return iou
