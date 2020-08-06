import cv2
import numpy as np
import os
import time

from math import ceil

class Extractor(object):
    def __init__(self, IECore, use_movidius=False):

        self.ie = IECore
        model_xml = "Openvino/person-reidentification-retail-0270.xml"
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        self.model = self.ie.read_network(model=model_xml, weights=model_bin)
        self.batch_size = 8
        self.model.batch_size = self.batch_size

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

        # Verify the number of output features
        self.n_feat = self.model.outputs[self.out_blob].shape[-1]

        # Make model executable
        # start = time.time()
        self.model = self.ie.load_network(network=self.model, device_name=self.device)
        # print("Load net time: {}s".format(time.time() - start))

    def _preprocess(self, im_crops):

        images = np.ndarray(shape=(len(im_crops), self.c, self.h, self.w))
        for i in range(len(im_crops)):
            # See input specs here:
            # https://docs.openvinotoolkit.org/latest/omz_models_intel_person_reidentification_retail_0270_description_person_reidentification_retail_0270.html

            # The identifier expects a BGR image, but the function input is RGB
            image = cv2.cvtColor(im_crops[i], cv2.COLOR_RGB2BGR)
            ih, iw = image.shape[:-1]
            if (ih, iw) != (self.h, self.w):
                image = cv2.resize(image, (self.w, self.h))
            image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            images[i] = image
        return images

    def __call__(self, im_crops):
        # Pass the blob through the network and obtain the detections and predictions
        data = self._preprocess(im_crops)
        # Fixed batch size:
        # features = self.model.infer(inputs={self.input_name: data})[self.out_blob]
        # For dynamic batch size, slow performance:
        n_batches = ceil(len(data) / self.batch_size)
        if n_batches * self.batch_size > data.shape[0]:
            new_data = np.zeros((n_batches * self.batch_size - len(data), self.c, self.h, self.w))
            data = np.concatenate((data, new_data), axis=0)
        features = np.ndarray((len(data), self.n_feat))
        for b in range(n_batches):
            batch = data[self.batch_size * b:self.batch_size * (b+1)]
            # start = time.time()
            local_feat = self.model.infer(inputs={self.input_name: batch})[self.out_blob][0]
            # print("Elapsed batch time: {}s".format(time.time()-start))
            features[self.batch_size * b:self.batch_size * (b+1), :] = local_feat
        return features[:len(im_crops), :]


        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()


if __name__ == '__main__':
    img = cv2.imread("demo.jpg")[:,:,(2,1,0)]
    extr = Extractor("checkpoint/ckpt.t7")
    feature = extr(img)
    print(feature.shape)

