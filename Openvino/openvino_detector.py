# Based on:
# https://github.com/openvinotoolkit/open_model_zoo/tree/master/demos/python_demos/object_detection_demo_ssd_async
# Using models from:
# https://github.com/openvinotoolkit/open_model_zoo/tree/master/demos/pedestrian_tracker_demo

import numpy as np
import cv2
import os
import time

from openvino.inference_engine import IECore


class Detector(object):

    def __init__(self, use_movidius=False):
        self.ie = IECore()
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
        confidences = np.zeros((n_detections,), dtype=np.float32)
        boxes = np.zeros((n_detections, 4), dtype=np.int)
        labels = np.zeros((n_detections,), dtype=np.int)

        for i, proposal in enumerate(detections):
            if proposal[2] > 0:
                labels[i] = 0
                confidences[i] = proposal[2]
                startX = np.int(iw * proposal[3])
                startY = np.int(ih * proposal[4])
                endX = np.int(iw * proposal[5])
                endY = np.int(ih * proposal[6])
                boxes[i, ...] = int((startX + endX) / 2), int((startY + endY) / 2), endX - startX, endY - startY

        not_dummy = np.where(confidences != 0)
        # print("Prediction time: {}s\n".format(time.time() - start))
        return boxes[not_dummy], confidences[not_dummy], labels[not_dummy]
