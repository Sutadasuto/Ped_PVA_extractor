import cv2
import numpy as np
import os
from math import ceil

COLORS_10 = [(144, 238, 144), (178, 34, 34), (221, 160, 221), (0, 255, 0), (0, 128, 0), (210, 105, 30), (220, 20, 60),
             (192, 192, 192), (255, 228, 196), (50, 205, 50), (139, 0, 139), (100, 149, 237), (138, 43, 226),
             (238, 130, 238),
             (255, 0, 255), (0, 100, 0), (127, 255, 0), (255, 0, 255), (0, 0, 205), (255, 140, 0), (255, 239, 213),
             (199, 21, 133), (124, 252, 0), (147, 112, 219), (106, 90, 205), (176, 196, 222), (65, 105, 225),
             (173, 255, 47),
             (255, 20, 147), (219, 112, 147), (186, 85, 211), (199, 21, 133), (148, 0, 211), (255, 99, 71),
             (144, 238, 144),
             (255, 255, 0), (230, 230, 250), (0, 0, 255), (128, 128, 0), (189, 183, 107), (255, 255, 224),
             (128, 128, 128),
             (105, 105, 105), (64, 224, 208), (205, 133, 63), (0, 128, 128), (72, 209, 204), (139, 69, 19),
             (255, 245, 238),
             (250, 240, 230), (152, 251, 152), (0, 255, 255), (135, 206, 235), (0, 191, 255), (176, 224, 230),
             (0, 250, 154),
             (245, 255, 250), (240, 230, 140), (245, 222, 179), (0, 139, 139), (143, 188, 143), (255, 0, 0),
             (240, 128, 128),
             (102, 205, 170), (60, 179, 113), (46, 139, 87), (165, 42, 42), (178, 34, 34), (175, 238, 238),
             (255, 248, 220),
             (218, 165, 32), (255, 250, 240), (253, 245, 230), (244, 164, 96), (210, 105, 30)]


def draw_bbox(img, box, cls_name, identity=None, offset=(0, 0)):
    """
        draw box of an id
    """
    x1, y1, x2, y2 = [int(i + offset[idx % 2]) for idx, i in enumerate(box)]
    # set color and label text
    color = COLORS_10[identity % len(COLORS_10)] if identity is not None else COLORS_10[0]
    label = '{} {}'.format(cls_name, identity)
    # box text and bar
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
    cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 1)
    return img


def draw_bboxes(detector, img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        # im_width = detector.im_width
        im_height = detector.im_height
        # display_width = detector.display_width
        display_height = detector.display_height
        font_scale = 2 * im_height / display_height
        font_thickness = ceil(2 * im_height / display_height)
        bbox_thickness = int(3 * im_height / display_height)
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = COLORS_10[id % len(COLORS_10)]
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, font_scale, font_thickness)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, bbox_thickness)
        cv2.rectangle(img, (x1, y1 - (t_size[1] + 4)), (x1 + t_size[0] + 3, y1), color, -1)
        cv2.putText(img, label, (x1, y1), cv2.FONT_HERSHEY_PLAIN, font_scale, [255, 255, 255], font_thickness)
    return img


def softmax(x):
    assert isinstance(x, np.ndarray), "expect x be a numpy array"
    x_exp = np.exp(x * 5)
    return x_exp / x_exp.sum()


def softmin(x):
    assert isinstance(x, np.ndarray), "expect x be a numpy array"
    x_exp = np.exp(-x)
    return x_exp / x_exp.sum()


def text_to_npy(database_text_files_dir):
    videos = sorted([os.path.join(database_text_files_dir, f) for f in os.listdir(database_text_files_dir) if
                     (os.path.isfile(os.path.join(database_text_files_dir, f)) or os.path.isdir(
                         os.path.join(database_text_files_dir, f))) and not f.startswith('.')
                     and not f.endswith(('~', 'gt'))], key=lambda f: f.lower())
    num_coordinates = 2
    video_file_arrays = []
    total_subjects_count = 0
    subjects_dict = ""
    for video in videos:
        video_id = os.path.split(video)[-1]
        print("CURRENT VIDEO: %s" % video)
        text_files = sorted([os.path.join(video, f) for f in os.listdir(video) if
                             (os.path.isfile(os.path.join(video, f)) or os.path.isdir(
                                 os.path.join(video, f))) and not f.startswith('.') and f.endswith('.txt')
                             and not f.endswith('~')], key=lambda f: f.lower())
        num_point_features = len(text_files)
        with open(text_files[0], 'r') as f:
            num_frames = len(f.readlines()) - 1  # first line is a comment explaining the format of the file
        video_file_array = np.array([], dtype=np.float).reshape(0, num_frames, num_coordinates * num_point_features)

        for file_num, text_file in enumerate(text_files):
            f = open(text_file, 'r')
            frames = f.readlines()
            for frame in frames:
                if frame.startswith("#"):
                    continue
                frame = frame.strip()
                frame_index = int(frame.split(',')[0].split('_')[-1]) - 1
                subjects = frame.split(',')[1:]
                if subjects == ['']:
                    continue
                for subject in subjects:
                    data = subject.split(' ')
                    id = int(data[0].split('_id_')[-1])
                    point_values = [float(value) for value in data[1:]]
                    if id + 1 > video_file_array.shape[0]:
                        new_columns = (id + 1) - video_file_array.shape[0]
                        new_columns = np.zeros((new_columns, num_frames, num_coordinates * num_point_features))
                        video_file_array = np.concatenate((video_file_array, new_columns), axis=0)
                    for i in range(num_coordinates):
                        video_file_array[id, frame_index, num_coordinates * file_num + i] = point_values[i]
        video_subject_ids = [i for i in range(video_file_array.shape[0])]
        subjects_dict += "{}\n".format("\n".join(["subject%s_video%s_trackID%s" % (id + total_subjects_count, video_id, id) for id in video_subject_ids]))
        total_subjects_count += len(video_subject_ids)
        video_file_arrays.append(video_file_array)
    final_array = np.concatenate(tuple(video_file_arrays), axis=0)
    np.save(database_text_files_dir, final_array)
    with open("%s.csv" % database_text_files_dir, 'w') as csv_dict:
        csv_dict.write(subjects_dict)


if __name__ == '__main__':
    # text_to_npy(
    #     "/media/winbuntu/google-drive/ESIEE/Master Stay/Databases/UCSD_Anomaly_Dataset.v1p2_processed/UCSDped1/Train")
    x = np.arange(10) / 10.
    x = np.array([0.5, 0.5, 0.5, 0.6, 1.])
    y = softmax(x)
    z = softmin(x)
    import ipdb;

    ipdb.set_trace()
