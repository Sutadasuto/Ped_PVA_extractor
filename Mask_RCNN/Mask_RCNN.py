# import necessary libraries
from PIL import Image
import matplotlib.pyplot as plt
import os
import time
import torch
import torchvision.transforms as T
import torchvision
import torch
import numpy as np
import cv2
import random


class Detector(object):

    def __init__(self, use_cuda=True):
        # get the pretrained model from torchvision.models
        # Note: pretrained=True will get the pretrained weights for the model.
        # model.eval() to use the model for inference
        self.use_cuda = use_cuda
        self.device = "cuda" if use_cuda else "cpu"
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.model.to(self.device)
        self.class_names = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

    def __call__(self, img):
        """
                get_prediction
                  parameters:
                    - rgb_img - rgb numpy array image
                  method:
                    - the image is converted to image tensor using PyTorch's Transforms
                    - image is passed through the model to get the predictions
                    - masks, classes and bounding boxes are obtained from the model and soft masks are made binary(0 or 1) on masks
                      ie: eg. segment of cat is made 1 and rest of the image is made 0

                """
        transform = T.Compose([T.ToTensor()])
        img = transform(img).to(self.device)

        pred = self.model([img])
        pred_score = pred[0]['scores'].cpu().detach().numpy()
        # pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
        masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
        pred_class = pred[0]['labels'].cpu().numpy()
        pred_boxes = [list(box) for box in pred[0]['boxes'].cpu().detach().numpy()]
        pred_boxes = [[(box[2]+box[0])/2.0, (box[3]+box[1])/2.0, box[2]-box[0], box[3]-box[1]] for box in pred_boxes]
        pred_boxes = np.array(pred_boxes)

        return pred_boxes, pred_score, pred_class, masks

    # def random_colour_masks(self, image):
    #     """
    #     random_colour_masks
    #       parameters:
    #         - image - predicted masks
    #       method:
    #         - the masks of each predicted object is given random colour for visualization
    #     """
    #     colours = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255], [80, 70, 180],
    #                [250, 80, 190], [245, 145, 50], [70, 150, 250], [50, 190, 190]]
    #     r = np.zeros_like(image).astype(np.uint8)
    #     g = np.zeros_like(image).astype(np.uint8)
    #     b = np.zeros_like(image).astype(np.uint8)
    #     r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0, 10)]
    #     coloured_mask = np.stack([r, g, b], axis=2)
    #     return coloured_mask

    # def get_prediction(self, img_path, threshold):
    #     """
    #     get_prediction
    #       parameters:
    #         - img_path - path of the input image
    #       method:
    #         - Image is obtained from the image path
    #         - the image is converted to image tensor using PyTorch's Transforms
    #         - image is passed through the model to get the predictions
    #         - masks, classes and bounding boxes are obtained from the model and soft masks are made binary(0 or 1) on masks
    #           ie: eg. segment of cat is made 1 and rest of the image is made 0
    #
    #     """
    #     real_start = time.time()
    #     img = Image.open(img_path)
    #     transform = T.Compose([T.ToTensor()])
    #     img = transform(img)
    #     if cuda:
    #         img = img.cuda()
    #     real_start = time.time()
    #     start = time.time()
    #     pred = model([img])
    #     end = time.time()
    #     print("Prediction time: %ss" % (end - start))
    #     pred_score = list(pred[0]['scores'].cpu().detach().numpy())
    #     pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    #     masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
    #     pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    #     pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].cpu().detach().numpy())]
    #     masks = masks[:pred_t + 1]
    #     pred_boxes = pred_boxes[:pred_t + 1]
    #     pred_class = pred_class[:pred_t + 1]
    #     real_end = time.time()
    #     print("Whole processing time: %ss" % (real_end - real_start))
    #     return masks, pred_boxes, pred_class

    # def instance_segmentation_api(self, img_path, threshold=0.5, rect_th=3, text_size=3, text_th=3):
    #     """
    #     instance_segmentation_api
    #       parameters:
    #         - img_path - path to input image
    #       method:
    #         - prediction is obtained by get_prediction
    #         - each mask is given random color
    #         - each mask is added to the image in the ration 1:0.8 with opencv
    #         - final output is displayed
    #     """
    #     masks, boxes, pred_cls = get_prediction(img_path, threshold)
    #     img = cv2.imread(img_path)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     for i in range(len(masks)):
    #         rgb_mask = random_colour_masks(masks[i])
    #         img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
    #         cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 255, 0), thickness=rect_th)
    #         cv2.putText(img, pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)
    #     plt.figure(figsize=(20, 30))
    #     plt.imshow(img)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.show()

    # instance_segmentation_api('images/mrcnn_standing_people.jpg', 0.75)