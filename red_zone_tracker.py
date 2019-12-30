import os
import cv2
import time
import argparse
import numpy as np
import re
from distutils.util import strtobool

from YOLOv3 import YOLOv3
from deep_sort import DeepSort
from demo_yolo3_deepsort import Detector
from util import DeviceVideoStream, draw_bboxes


class ZoneDetector(Detector):

    def __init__(self, args):
        super().__init__(args)
        self.m = None
        self.b = None
        self.zone = None
        self.mouse_coordinates = []
        self.red_zone_defined = False

    def define_red_zone(self, image, choice):
        if type(choice) is tuple:
            self.m, self.b, self.zone = choice
            if type(self.m) is not float or type(self.b) is not float or (self.zone != "up" or self.zone != "down"):
                raise ValueError("Expected format is (float, float, str), where the str must be either 'up' or 'down'.")
            self.red_zone_defined = True
        elif choice == "draw":
            if type(image) == DeviceVideoStream:
                cv2.namedWindow("Sample image")
                cv2.setMouseCallback("Sample image", self.select_line)

                # keep looping until the 'q' key is pressed
                while True:
                    # display the image and wait for a keypress
                    new_image, _ = image.read()

                    if len(self.mouse_coordinates) == 3:
                        x1, y1, x2, y2 = [value for pair in self.mouse_coordinates[:-1] for value in pair]
                        try:
                            self.m = (y2 - y1) / (x2 - x1)
                        except ZeroDivisionError:
                            self.m = 1e10
                        if self.m == 0:
                            self.m = 1e-10
                        self.b = y1 - self.m * x1
                        new_image = self.draw_red_zone(new_image)
                        if self.mouse_coordinates[-1][-1] >= self.mouse_coordinates[-1][0] * self.m + self.b:
                            self.zone = "up"
                        else:
                            self.zone = "down"

                    cv2.imshow("Sample image", new_image)
                    key = cv2.waitKey(1) & 0xFF

                    # if the 'r' key is pressed, reset the cropping region
                    if key == ord("r"):
                        self.mouse_coordinates = []

                    # if the 'q' key is pressed, break from the loop
                    elif key == ord("q") and len(self.mouse_coordinates) == 3:
                        cv2.destroyWindow("Sample image")
                        self.red_zone_defined = True
                        break

            else:
                clone = image.copy()
                cv2.namedWindow("Sample image")
                cv2.setMouseCallback("Sample image", self.select_line)

                # keep looping until the 'q' key is pressed
                while True:
                    # display the image and wait for a keypress
                    cv2.imshow("Sample image", image)
                    key = cv2.waitKey(1) & 0xFF

                    if len(self.mouse_coordinates) == 3:
                        x1, y1, x2, y2 = [value for pair in self.mouse_coordinates[:-1] for value in pair]
                        self.m = (y2 - y1) / (x2 - x1)
                        if self.m == 0:
                            self.m = 1e-10
                        self.b = y1 - self.m * x1
                        image = self.draw_red_zone(image)
                        if self.mouse_coordinates[-1][-1] >= self.mouse_coordinates[-1][0] * self.m + self.b:
                            self.zone = "up"
                        else:
                            self.zone = "down"

                        # if the 'r' key is pressed, reset the cropping region
                    if key == ord("r"):
                        image = clone.copy()
                        self.mouse_coordinates = []

                    # if the 'q' key is pressed, break from the loop
                    elif key == ord("q") and len(self.mouse_coordinates) == 3:
                        cv2.destroyWindow("Sample image")
                        self.red_zone_defined = True
                        break

    def draw_red_zone(self, image):
        thickness = int(self.im_height / 100)
        line_p1 = (int(-self.b / self.m), 0)
        line_p2 = (int((self.im_height - 1 - self.b) / self.m), self.im_height - 1)
        image = cv2.line(image, line_p1, line_p2, (0, 0, 255), thickness)
        image = cv2.circle(image, self.mouse_coordinates[-1], 2*thickness, (0, 0, 255), thickness)
        return image

    def select_line(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.mouse_coordinates) < 3:
            self.mouse_coordinates.append((x, y))

    def detect_per_zone(self):

        # Mass centers
        mc = [np.array([], dtype=np.float).reshape(2, 0, 3) for i in range(2)]  # [[[x_t-1,y_t-1,frame_num_t-1]],
        # [[x_t,y_t,frame_num_t]]]. One array for store positions, one for velocities
        # Top-left corner of bounding boxes
        tl = [np.array([], dtype=np.float).reshape(2, 0, 3) for i in range(2)]
        # Top-right
        tr = [np.array([], dtype=np.float).reshape(2, 0, 3) for i in range(2)]
        bl = [np.array([], dtype=np.float).reshape(2, 0, 3) for i in range(2)]
        br = [np.array([], dtype=np.float).reshape(2, 0, 3) for i in range(2)]

        analyzed_points = [mc, tl, tr, bl, br]
        analyzed_points_dict = {
            0: "mc", 1: "tl", 2: "tr", 3: "bl", 4: "br"
        }
        self.open_text_files(analyzed_points_dict)

        counter = self.sampling_delay
        real_frame = 0
        while True:
            start = time.time()
            print("Source fps: %s" % self.source_fps)
            if not self.using_camera:
                grabbed, ori_im = self.vdo.read()
                if not grabbed:
                    break
                if not self.red_zone_defined:
                    self.define_red_zone(ori_im, "draw")
            else:
                if not self.red_zone_defined:
                    self.define_red_zone(self.stream, "draw")
                ori_im, buffered_frames = self.stream.read()

            if counter != self.sampling_delay:
                counter += 1
                continue

            counter = 1
            real_frame += 1

            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
            im = ori_im
            bbox_xcycwh, cls_conf, cls_ids = self.yolo3(im)

            if not self.using_camera:
                self.frame_index += 1
            else:
                self.frame_index += buffered_frames
            frame_strs = [["frame_%s," % real_frame for i in range(len(self.outputs_dict[0]))] for j in
                          range(len(self.outputs_dict.keys()))]

            if bbox_xcycwh is not None:
                # select class person
                mask = cls_ids == 0

                bbox_xcycwh = bbox_xcycwh[mask]
                # bbox_xcycwh[:,3:] *= 1.2

                cls_conf = cls_conf[mask]
                outputs = self.deepsort.update(bbox_xcycwh, cls_conf, im)
                ori_im = self.draw_red_zone(ori_im)
                if len(outputs) > 0:
                    # Expand the memory arrays size if more subjects are tracked
                    for point in range(len(analyzed_points)):
                        for feature in range(len(analyzed_points[point])):
                            analyzed_points[point][feature] = \
                                self.add_subjects(outputs[:, -1], analyzed_points[point][feature])

                    outputs[:, -1] -= 1
                    bottom_centers = np.array([
                        [int((subject[0] + subject[2]) / 2), subject[3]] for subject in outputs
                    ])
                    if self.zone == "up":
                        red_indices = np.where(bottom_centers[:, 0] * self.m + self.b < bottom_centers[:, 1])
                    else:
                        red_indices = np.where(bottom_centers[:, 0] * self.m + self.b > bottom_centers[:, 1])

                    mass_centers = np.array([
                        [int((subject[0] + subject[2]) / 2), int((subject[1] + subject[3]) / 2)] for subject in outputs
                    ])
                    tl_corners = outputs[:, [0, 1]]
                    br_corners = outputs[:, [2, 3]]
                    tr_corners = np.concatenate((br_corners[:, 0][:, None], tl_corners[:, 1][:, None]), axis=1)
                    bl_corners = np.concatenate((tl_corners[:, 0][:, None], br_corners[:, 1][:, None]), axis=1)

                    mass_centers = mass_centers[red_indices]
                    tl_corners = tl_corners[red_indices]
                    br_corners = br_corners[red_indices]
                    tr_corners = tr_corners[red_indices]
                    bl_corners = bl_corners[red_indices]
                    analyzed_points_current_values = [mass_centers, tl_corners, tr_corners, bl_corners, br_corners]

                    bbox_xyxy = outputs[:, :4][red_indices]
                    identities = outputs[:, 4][red_indices]
                    red_zone_feet = bottom_centers[red_indices]

                    for i in range(len(analyzed_points)):
                        frame_strs[i][0], frame_strs[i][1], frame_strs[i][2] = self.feat_to_str(
                            analyzed_points_current_values[i],
                            analyzed_points[i][0],
                            identities, analyzed_points[i][1],
                            frame_strs[i][0], frame_strs[i][1], frame_strs[i][2])

                    ori_im = draw_bboxes(self, ori_im, bbox_xyxy, identities)
                    for point in red_zone_feet:
                        cv2.circle(ori_im, tuple(point), int(self.im_height / 100), (0, 0, 255), -1)
            else:
                ori_im = self.draw_red_zone(ori_im)

            for i in range(len(frame_strs)):
                for j in range(len(frame_strs[i])):
                    self.outputs_dict[i][j].write("%s\n" % frame_strs[i][j])

            end = time.time()
            print("time: {:.3f}s, fps: {:.1f}, processed frames: {}".format(end - start, 1 / (end - start), real_frame))

            if not bool(strtobool(self.args.ignore_display)):
                cv2.imshow("test", ori_im)
                # cv2.waitKey(1)

            if self.args.save_path is not None:
                self.output.write(ori_im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if self.using_camera:
            self.stream.stop()
        self.close_text_files()


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--frame_rate", type=float, default=0)
    parser.add_argument("--yolo_cfg", type=str, default="YOLOv3/cfg/yolo_v3.cfg")
    parser.add_argument("--yolo_weights", type=str, default="YOLOv3/yolov3.weights")
    parser.add_argument("--yolo_names", type=str, default="YOLOv3/cfg/coco.names")
    parser.add_argument("--conf_thresh", type=float, default=0.5)
    parser.add_argument("--nms_thresh", type=float, default=0.4)
    parser.add_argument("--deepsort_checkpoint", type=str, default="deep_sort/deep/checkpoint/ckpt.t7")
    parser.add_argument("--max_dist", type=float, default=0.2)
    parser.add_argument("--ignore_display", type=str, default="False")
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="demo.avi")
    parser.add_argument("--use_cuda", type=str, default="True")
    return parser.parse_args(args)


def main(args):
    with ZoneDetector(args) as det:
        det.detect_per_zone()


if __name__ == "__main__":
    args = parse_args()
    main(args)