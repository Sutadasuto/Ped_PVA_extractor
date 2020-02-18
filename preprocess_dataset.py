import argparse
import os

from util import text_to_npy

from demo_yolo3_deepsort import Tracker
from distutils.util import strtobool


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    # User intended
    parser.add_argument("DATABASE_FOLDER", type=str)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--frame_rate", type=float, default=0)
    parser.add_argument("--detector", type=str, default="yolov3")
    parser.add_argument("--conf_thresh", type=float, default=0.5)
    parser.add_argument("--nms_thresh", type=float, default=0.4)
    parser.add_argument("--max_dist", type=float, default=0.2)
    parser.add_argument("--max_age", type=int, default=70)
    parser.add_argument("--save_videos", type=bool, default=False)
    parser.add_argument("--convert_txt2npy", type=bool, default=False)
    parser.add_argument("--ignore_display", type=str, default="True")
    parser.add_argument("--use_cuda", type=str, default="True")

    # Not user intended
    parser.add_argument("--VIDEO_PATH", type=str)
    parser.add_argument("--yolo_cfg", type=str, default="YOLOv3/cfg/yolo_v3.cfg")
    parser.add_argument("--yolo_weights", type=str, default="YOLOv3/yolov3.weights")
    parser.add_argument("--yolo_names", type=str, default="YOLOv3/cfg/coco.names")
    parser.add_argument("--deepsort_checkpoint", type=str, default="deep_sort/deep/checkpoint/ckpt.t7")
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="bboxes.avi")

    return parser.parse_args(args)


def main(args):
    if not os.path.isdir(args.DATABASE_FOLDER):
        raise NotADirectoryError
    videos = sorted([os.path.join(args.DATABASE_FOLDER, f) for f in os.listdir(args.DATABASE_FOLDER) if
                     (os.path.isfile(os.path.join(args.DATABASE_FOLDER, f)) or os.path.isdir(
                         os.path.join(args.DATABASE_FOLDER, f))) and not f.startswith('.')
                            and not f.endswith('~')], key=lambda f: f.lower())
    if args.output_dir:
        output_root_dir = args.output_dir
    else:
        output_root_dir = os.path.join(os.getcwd(), "outputs")
    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir)

    if args.save_videos:
        video_name = args.save_path

    for video in videos:
        print("\n\n***\nCURRENT VIDEO: %s\n" % video)
        args.VIDEO_PATH = video
        args.output_dir = os.path.join(output_root_dir, os.path.split(video)[-1])
        if args.save_videos:
            args.save_path = os.path.join(args.output_dir, video_name)
        else:
            args.save_path = None
        with Tracker(args) as det:
            det.detect()
    if args.convert_txt2npy:
        text_to_npy(output_root_dir, os.path.split(args.DATABASE_FOLDER)[-1])


if __name__ == "__main__":
    args = parse_args()
    main(args)
