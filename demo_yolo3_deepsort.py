import os
import cv2
import time
import argparse
import numpy as np
import re
from distutils.util import strtobool

from YOLOv3 import YOLOv3
from deep_sort import DeepSort
from util import draw_bboxes


class Detector(object):
    def __init__(self, args):
        self.args = args
        use_cuda = bool(strtobool(self.args.use_cuda))
        if not args.ignore_display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)
        self.display_width = args.display_width
        self.display_height = args.display_height

        self.vdo = cv2.VideoCapture()
        self.yolo3 = YOLOv3(args.yolo_cfg, args.yolo_weights, args.yolo_names, is_xywh=True,
                            conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh, use_cuda=use_cuda)
        self.deepsort = DeepSort(args.deepsort_checkpoint, use_cuda=use_cuda)
        self.class_names = self.yolo3.class_names
        if args.output_dir is None:
            self.output_dir = os.path.join(os.getcwd(), "outputs")
        else:
            self.output_dir = args.output_dir

    def __enter__(self):
        self.open_stream()
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.args.frame_rate > 0:
            if self.args.frame_rate <= self.source_fps:
                self.sampling_delay = int(self.source_fps / self.args.frame_rate)  # frames
            else:
                raise ValueError("Frame rate can't be greater than source: %sfps" % self.source_fps)
            self.frame_period = 1.0 / self.args.frame_rate  # seconds
        else:
            self.sampling_delay = 1
            self.frame_period = 1.0 / self.source_fps  # seconds
            self.args.frame_rate = self.source_fps
        self.frame_index = 0

        if self.args.save_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            video_dir = os.path.split(self.args.save_path)[0]
            if not os.path.exists(video_dir) and len(video_dir) > 0:
                os.makedirs(video_dir)
            self.output = cv2.VideoWriter(self.args.save_path, fourcc, self.args.frame_rate,
                                          (self.im_width, self.im_height))

        assert self.vdo.isOpened()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def detect(self):

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
        while self.vdo.grab():
            start = time.time()
            _, ori_im = self.vdo.retrieve()
            if counter != self.sampling_delay:
                counter += 1
                continue
            counter = 1
            real_frame += 1

            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
            im = ori_im
            bbox_xcycwh, cls_conf, cls_ids = self.yolo3(im)

            self.frame_index += 1
            frame_strs = [["frame_%s," % self.frame_index for i in range(len(self.outputs_dict[0]))] for j in
                          range(len(self.outputs_dict.keys()))]

            if bbox_xcycwh is not None:
                # select class person
                mask = cls_ids == 0

                bbox_xcycwh = bbox_xcycwh[mask]
                # bbox_xcycwh[:,3:] *= 1.2

                cls_conf = cls_conf[mask]
                outputs = self.deepsort.update(bbox_xcycwh, cls_conf, im)
                if len(outputs) > 0:
                    # Expand the memory arrays size if more subjects are tracked
                    for point in range(len(analyzed_points)):
                        for feature in range(len(analyzed_points[point])):
                            analyzed_points[point][feature] = \
                                self.add_subjects(outputs[:, -1], analyzed_points[point][feature])

                    outputs[:, -1] -= 1
                    mass_centers = np.array([
                        [int((subject[0] + subject[2]) / 2), int((subject[1] + subject[3]) / 2)] for subject in outputs
                    ])
                    tl_corners = outputs[:, [0, 1]]
                    br_corners = outputs[:, [2, 3]]
                    tr_corners = np.concatenate((br_corners[:, 0][:, None], tl_corners[:, 1][:, None]), axis=1)
                    bl_corners = np.concatenate((tl_corners[:, 0][:, None], br_corners[:, 1][:, None]), axis=1)
                    analyzed_points_current_values = [mass_centers, tl_corners, tr_corners, bl_corners, br_corners]
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, 4]

                    for i in range(len(analyzed_points)):
                        frame_strs[i][0], frame_strs[i][1], frame_strs[i][2] = self.feat_to_str(
                            analyzed_points_current_values[i],
                            analyzed_points[i][0],
                            identities, analyzed_points[i][1],
                            frame_strs[i][0], frame_strs[i][1], frame_strs[i][2])

                    ori_im = draw_bboxes(self, ori_im, bbox_xyxy, identities)

            for i in range(len(frame_strs)):
                for j in range(len(frame_strs[i])):
                    self.outputs_dict[i][j].write("%s\n" % frame_strs[i][j])

            end = time.time()
            print("time: {:.3f}s, fps: {:.1f}, frame_number: {}".format(end - start, 1 / (end - start), real_frame))

            if not self.args.ignore_display:
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)

            if self.args.save_path is not None:
                self.output.write(ori_im)
        self.close_text_files()

    def add_subjects(self, frame_identities, array_to_complete):
        if frame_identities.max() > array_to_complete.shape[1]:
            additional_subjects = frame_identities.max() - array_to_complete.shape[1]
            complementary_array = np.array([
                [[-1, -1, -1] for i in range(additional_subjects)],
                [[-1, -1, -1] for j in range(additional_subjects)]
            ])
            return np.concatenate((array_to_complete, complementary_array), axis=1)
        return array_to_complete

    def close_text_files(self):
        for key in list(self.outputs_dict.keys()):
            for file in self.outputs_dict[key]:
                file.close()

    def derivate(self, current_variable_values, variables_per_sub_history, identities):
        out_str = ""
        current_derivative_state = []
        for identity in identities:
            values = variables_per_sub_history[:, identity, :]
            if values[0, -1] == values[1, -1]:
                variables_per_sub_history[1, identity, :] = np.concatenate(
                    (current_variable_values[list(identities).index(identity)], [self.frame_index])
                )

            frames = variables_per_sub_history[:, identity, -1]
            if frames[1] > frames[0]:
                xs = variables_per_sub_history[:, identity, 0]
                ys = variables_per_sub_history[:, identity, 1]

                elapsed_time = (frames[1] - frames[0]) * self.frame_period
                d_x = (xs[1] - xs[0]) / elapsed_time
                d_y = (ys[1] - ys[0]) / elapsed_time
                if frames[0] > -1:  # -1 stands for initialization frame, not contained in video
                    out_str += "track_id_%s %.1f %.1f," % (identity, d_x, d_y)
                    current_derivative_state.append([d_x, d_y, identity])

                variables_per_sub_history[0, identity, :] = np.concatenate(
                    (current_variable_values[list(identities).index(identity)], [self.frame_index])
                )
        return out_str, np.array(current_derivative_state)

    def extract_features(self, current_variable_values, variables_per_sub_history, current_identities,
                         derivatives_per_sub_history):
        d_str, outputs_d = self.derivate(current_variable_values, variables_per_sub_history, current_identities)

        if len(outputs_d) > 0:
            dd_str, outputs_dd = self.derivate(
                outputs_d[:, :-1], derivatives_per_sub_history, outputs_d[:, -1].astype(int)
            )
        else:
            dd_str = ","

        return d_str[:-1], dd_str[:-1]

    def feat_to_str(self, current_variable_values, variables_per_sub_history, current_identities,
                    derivatives_per_sub_history, var_string, dvar_string, ddvar_string):
        o_str = ",".join([
            "track_id_%s %s" % (identity, " ".join(['%.1f' % value for value in values])) for identity, values in
            zip(current_identities, current_variable_values)
        ])
        d_str, dd_str = self.extract_features(
            current_variable_values, variables_per_sub_history, current_identities, derivatives_per_sub_history
        )
        var_string += o_str
        dvar_string += d_str
        ddvar_string += dd_str

        return var_string, dvar_string, ddvar_string

    def open_stream(self):
        if os.path.isfile(self.args.VIDEO_PATH):
            try:
                self.vdo.open(self.args.VIDEO_PATH)
            except IOError:
                raise IOError("%s is not a valid video file." % self.args.VIDEO_PATH)
            self.source_fps = self.vdo.get(cv2.CAP_PROP_FPS)
        elif os.path.isdir(self.args.VIDEO_PATH):
            if self.args.frame_rate == 0:
                self.source_fps = 29.7
            else:
                self.source_fps = self.args.frame_rate
            format_str = sorted([f for f in os.listdir(self.args.VIDEO_PATH)
                                 if os.path.isfile(os.path.join(self.args.VIDEO_PATH, f))
                                 and not f.startswith('.') and not f.endswith('~')], key=lambda f: f.lower())[0]
            numeration = re.findall('[0-9]+', format_str)
            len_num = len(numeration[-1])
            format_str = format_str.replace(numeration[-1], "%0{}d".format(len_num))
            self.vdo.open(os.path.join(self.args.VIDEO_PATH, format_str))
        else:
            raise ValueError("{} is neither a valid video file nor a folder with valid images.".format(self.args.VIDEO_PATH))

    def open_text_files(self, analyzed_points_dict):
        self.outputs_dict = {}
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        variables = ['pos', 'vel', 'acc']
        for key in list(analyzed_points_dict.keys()):
            self.outputs_dict[key] = [
                open(os.path.join(self.output_dir, 'out_%s_%s.txt' % (analyzed_points_dict[key], variable)), 'w') for
                variable in variables
            ]

        help_str = "frame_number,tracked_person_m_id *ph*_x_m *ph*_y_m,tracked_person_n_id *ph*_x_n *ph*_y_n,..."

        for key in list(self.outputs_dict.keys()):
            for variable, file in enumerate(self.outputs_dict[key]):
                file.write("# %s\n" % help_str.replace('*ph*', variables[variable]))


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
    parser.add_argument("--ignore_display", type=bool, default=False)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="demo.avi")
    parser.add_argument("--use_cuda", type=str, default="True")
    return parser.parse_args(args)


def main(args):
    with Detector(args) as det:
        det.detect()


if __name__ == "__main__":
    args = parse_args()
    main(args)
