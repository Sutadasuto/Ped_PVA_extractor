import os
import cv2
import time
import argparse
import numpy as np
from distutils.util import strtobool
from demo_yolo3_deepsort import Detector
from math import ceil
from util import DeviceVideoStream, draw_bboxes


class ZoneDetector(Detector):

    def __init__(self, args):
        super().__init__(args)
        if args.red_line_txt is None:
            self.text_file_path = args.VIDEO_PATH.split(".")
            self.text_file_path[-1] = "txt"
            self.text_file_path = ".".join(self.text_file_path)
        else:
            self.text_file_path = args.red_line_txt

        self.ms, self.bs, self.zones = [], [], []
        self.m_threshold = 1.0
        self.red_zone_defined = False
        self.mouse_coordinates = self.read_line_from_text(self.text_file_path)
        self.track_point_position = args.track_point_position
        self.n_frames = args.frames_memory_size

    def calculate_zone_for_first_lines(self, middle_index, middle_index_x1, middle_index_x2):
        old_x1 = middle_index_x1
        old_x2 = middle_index_x2
        for i in reversed(range(middle_index)):
            old_dir = np.sign(old_x2 - old_x1)
            c_x1, c_x2 = [pair[0] for pair in self.mouse_coordinates[i:i+2]]
            c_dir = np.sign(c_x2 - c_x1)
            if c_dir == old_dir:
                if self.zones[i+1] == "up" or self.zones[i+1] == "down":
                    if abs(self.ms[i]) < self.m_threshold:
                        self.zones[i] = self.zones[i+1]
                    else:
                        shift = -1 * c_dir
                        test_x = c_x2 + shift
                        new_line_pos = np.sign(
                            (self.ms[i] * test_x + self.bs[i]) - (self.ms[i + 1] * test_x + self.bs[i + 1]))
                        if c_dir == 1:
                            if self.zones[i + 1] == "up":
                                if new_line_pos == 1:
                                    self.zones[i] = "right"
                                else:
                                    self.zones[i] = "left"
                            else:
                                if new_line_pos == 1:
                                    self.zones[i] = "left"
                                else:
                                    self.zones[i] = "right"
                        else:
                            if self.zones[i + 1] == "up":
                                if new_line_pos == 1:
                                    self.zones[i] = "left"
                                else:
                                    self.zones[i] = "right"
                            else:
                                if new_line_pos == 1:
                                    self.zones[i] = "right"
                                else:
                                    self.zones[i] = "left"
                else:
                    if abs(self.ms[i]) < self.m_threshold:
                        shift = c_dir
                        test_x = c_x2 + shift
                        new_line_pos = np.sign(
                            (self.ms[i] * test_x + self.bs[i]) - (self.ms[i + 1] * test_x + self.bs[i + 1]))
                        if c_dir == 1:
                            if new_line_pos == 1:
                                if self.zones[i+1] == "right":
                                    self.zones[i] = "up"
                                else:
                                    self.zones[i] = "down"
                            else:
                                if self.zones[i+1] == "right":
                                    self.zones[i] = "down"
                                else:
                                    self.zones[i] = "up"
                        else:
                            if new_line_pos == 1:
                                if self.zones[i+1] == "right":
                                    self.zones[i] = "down"
                                else:
                                    self.zones[i] = "up"
                            else:
                                if self.zones[i+1] == "right":
                                    self.zones[i] = "up"
                                else:
                                    self.zones[i] = "down"
                    else:
                        if np.sign(self.ms[i+1]) == np.sign(self.ms[i]):
                            self.zones[i] = self.zones[i+1]
                        else:
                            if self.zones[i+1] == "left":
                                self.zones[i] = "right"
                            else:
                                self.zones[i] = "left"
            else:
                if self.zones[i+1] == "up" or self.zones[i+1] == "down":
                    if abs(self.ms[i]) < self.m_threshold:
                        if self.zones[i+1] == "up":
                            self.zones[i] = "down"
                        else:
                            self.zones[i] = "up"
                    else:
                        shift = -1 * c_dir
                        test_x = c_x2 + shift
                        new_line_pos = np.sign(
                            (self.ms[i] * test_x + self.bs[i]) - (self.ms[i + 1] * test_x + self.bs[i + 1]))
                        if c_dir == 1:
                            if self.zones[i + 1] == "up":
                                if new_line_pos == 1:
                                    self.zones[i] = "left"
                                else:
                                    self.zones[i] = "right"
                            else:
                                if new_line_pos == 1:
                                    self.zones[i] = "right"
                                else:
                                    self.zones[i] = "left"
                        else:
                            if self.zones[i + 1] == "up":
                                if new_line_pos == 1:
                                    self.zones[i] = "right"
                                else:
                                    self.zones[i] = "left"
                            else:
                                if new_line_pos == 1:
                                    self.zones[i] = "left"
                                else:
                                    self.zones[i] = "right"
                else:
                    if abs(self.ms[i]) < self.m_threshold:
                        shift = -1 * c_dir
                        test_x = c_x2 + shift
                        new_line_pos = np.sign(
                            (self.ms[i] * test_x + self.bs[i]) - (self.ms[i + 1] * test_x + self.bs[i + 1]))
                        if c_dir == 1:
                            if new_line_pos == 1:
                                if self.zones[i + 1] == "right":
                                    self.zones[i] = "up"
                                else:
                                    self.zones[i] = "down"
                            else:
                                if self.zones[i + 1] == "right":
                                    self.zones[i] = "down"
                                else:
                                    self.zones[i] = "up"
                        else:
                            if new_line_pos == 1:
                                if self.zones[i + 1] == "right":
                                    self.zones[i] = "down"
                                else:
                                    self.zones[i] = "up"
                            else:
                                if self.zones[i + 1] == "right":
                                    self.zones[i] = "up"
                                else:
                                    self.zones[i] = "down"
                    else:
                        if np.sign(self.ms[i+1]) != np.sign(self.ms[i]):
                            self.zones[i] = self.zones[i+1]
                        else:
                            if self.zones[i+1] == "left":
                                self.zones[i] = "right"
                            else:
                                self.zones[i] = "left"
            old_x1 = c_x1
            old_x2 = c_x2

    def calculate_zone_for_last_lines(self, middle_index, middle_index_x1, middle_index_x2):
        old_x1 = middle_index_x1
        old_x2 = middle_index_x2
        for i in range(middle_index+1, len(self.ms)):
            old_dir = np.sign(old_x2 - old_x1)
            c_x1, c_x2 = [pair[0] for pair in self.mouse_coordinates[i:i + 2]]
            c_dir = np.sign(c_x2 - c_x1)
            if c_dir == old_dir:
                if self.zones[i - 1] == "up" or self.zones[i - 1] == "down":
                    if abs(self.ms[i]) < self.m_threshold:
                        self.zones[i] = self.zones[i-1]
                    else:
                        shift = c_dir
                        test_x = c_x1 + shift
                        new_line_pos = np.sign(
                            (self.ms[i] * test_x + self.bs[i]) - (self.ms[i-1] * test_x + self.bs[i-1]))
                        if c_dir == 1:
                            if self.zones[i-1] == "up":
                                if new_line_pos == 1:
                                    self.zones[i] = "left"
                                else:
                                    self.zones[i] = "right"
                            else:
                                if new_line_pos == 1:
                                    self.zones[i] = "right"
                                else:
                                    self.zones[i] = "left"
                        else:
                            if self.zones[i-1] == "up":
                                if new_line_pos == 1:
                                    self.zones[i] = "right"
                                else:
                                    self.zones[i] = "left"
                            else:
                                if new_line_pos == 1:
                                    self.zones[i] = "left"
                                else:
                                    self.zones[i] = "right"
                else:
                    if abs(self.ms[i]) < self.m_threshold:
                        shift = -1*c_dir
                        test_x = c_x1 + shift
                        new_line_pos = np.sign(
                            (self.ms[i] * test_x + self.bs[i]) - (self.ms[i - 1] * test_x + self.bs[i - 1]))
                        if c_dir == 1:
                            if self.zones[i-1] == "right":
                                if new_line_pos == 1:
                                    self.zones[i] = "down"
                                else:
                                    self.zones[i] = "up"
                            else:
                                if new_line_pos == 1:
                                    self.zones[i] = "up"
                                else:
                                    self.zones[i] = "down"
                        else:
                            if self.zones[i-1] == "right":
                                if new_line_pos == 1:
                                    self.zones[i] = "up"
                                else:
                                    self.zones[i] = "down"
                            else:
                                if new_line_pos == 1:
                                    self.zones[i] = "down"
                                else:
                                    self.zones[i] = "up"
                    else:
                        if np.sign(self.ms[i-1]) == np.sign(self.ms[i]):
                            self.zones[i] = self.zones[i-1]
                        else:
                            if self.zones[i-1] == "right":
                                self.zones[i] = "left"
                            else:
                                self.zones[i] = "right"
            else:
                if self.zones[i - 1] == "up" or self.zones[i - 1] == "down":
                    if abs(self.ms[i]) < self.m_threshold:
                        if self.zones[i-1] == "up":
                            self.zones[i] = "down"
                        else:
                            self.zones[i] = "up"
                    else:
                        shift = c_dir
                        test_x = c_x1 + shift
                        new_line_pos = np.sign(
                            (self.ms[i] * test_x + self.bs[i]) - (self.ms[i - 1] * test_x + self.bs[i - 1]))
                        if c_dir == 1:
                            if self.zones[i-1] == "up":
                                if new_line_pos == 1:
                                    self.zones[i] = "right"
                                else:
                                    self.zones[i] = "left"
                            else:
                                if new_line_pos == 1:
                                    self.zones[i] = "left"
                                else:
                                    self.zones[i] = "right"
                        else:
                            if self.zones[i-1] == "up":
                                if new_line_pos == 1:
                                    self.zones[i] = "left"
                                else:
                                    self.zones[i] = "right"
                            else:
                                if new_line_pos == 1:
                                    self.zones[i] = "right"
                                else:
                                    self.zones[i] = "left"
                else:
                    if abs(self.ms[i]) < self.m_threshold:
                        shift = c_dir
                        test_x = c_x1 + shift
                        new_line_pos = np.sign(
                            (self.ms[i] * test_x + self.bs[i]) - (self.ms[i - 1] * test_x + self.bs[i - 1]))
                        if c_dir == 1:
                            if self.zones[i-1] == "right":
                                if new_line_pos == 1:
                                    self.zones[i] = "down"
                                else:
                                    self.zones[i] = "up"
                            else:
                                if new_line_pos == 1:
                                    self.zones[i] = "up"
                                else:
                                    self.zones[i] = "down"
                        else:
                            if self.zones[i-1] == "right":
                                if new_line_pos == 1:
                                    self.zones[i] = "up"
                                else:
                                    self.zones[i] = "down"
                            else:
                                if new_line_pos == 1:
                                    self.zones[i] = "down"
                                else:
                                    self.zones[i] = "up"
                    else:
                        if np.sign(self.ms[i-1]) != np.sign(self.ms[i]):
                            self.zones[i] = self.zones[i-1]
                        else:
                            if self.zones[i-1] == "left":
                                self.zones[i] = "right"
                            else:
                                self.zones[i] = "left"
            old_x1 = c_x1
            old_x2 = c_x2

    def calculate_red_line(self):
        for start_index in range(len(self.mouse_coordinates) - 2):
            x1, y1, x2, y2 = [value for pair in self.mouse_coordinates[start_index:start_index + 2] for value in pair]
            if x1 == x2:
                x2 += 1
                self.mouse_coordinates[start_index + 1] = (x2, y2)
            try:
                self.ms.append((y2 - y1) / (x2 - x1))
            except ZeroDivisionError:
                self.ms.append(1e10)
            if self.ms[-1] == 0:
                self.ms[-1] = 1e-10
            self.bs.append(y1 - self.ms[-1] * x1)

        central_line, old_x1, old_x2 = self.find_first_zone()
        if central_line == -1:
            return False

        self.calculate_zone_for_first_lines(central_line, old_x1, old_x2)
        self.calculate_zone_for_last_lines(central_line, old_x1, old_x2)
        return True

    def define_red_zone(self, source_image):
        if len(self.mouse_coordinates) >= 3:
            self.red_zone_defined = self.calculate_red_line()
            if type(source_image) == DeviceVideoStream:
                image, _ = source_image.read()
            else:
                image = source_image.copy()
            image = self.draw_red_zone(image)
            if not self.red_zone_defined:
                cv2.imwrite(self.text_file_path.replace("txt", "jpg"), image)
                raise ValueError("The text file contains either an invalid format or a non-acceptable set of points. Non-acceptable set of points often involve an invalid or ambiguous ROI.")

        else:
            cv2.namedWindow("Sample image")
            cv2.setMouseCallback("Sample image", self.select_line)

            # keep looping until the 'q' key is pressed
            while True:
                # display the image and wait for a keypress
                if type(source_image) == DeviceVideoStream:
                    image, _ = source_image.read()
                else:
                    image = source_image.copy()

                if len(self.mouse_coordinates) >= 3:
                    image = self.draw_red_zone(image)

                cv2.imshow("Sample image", image)
                key = cv2.waitKey(1) & 0xFF

                # if the 'r' key is pressed, reset the cropping region
                if key == ord("r"):
                    self.mouse_coordinates = []

                # if the 'q' key is pressed, break from the loop
                elif key == ord("q") and len(self.mouse_coordinates) >= 3:
                    self.red_zone_defined = self.calculate_red_line()
                    if self.red_zone_defined:
                        cv2.destroyWindow("Sample image")
                        break
                    else:
                        print("Invalid ROI. Try a new click.")
                        print("Advices:\n*Use a point near a line segment\n*Use a point enclosed by line segments")
                        self.mouse_coordinates = self.mouse_coordinates[:-1]

        self.write_red_line_to_text(self.mouse_coordinates, self.text_file_path)
        cv2.imwrite(self.text_file_path.replace("txt", "jpg"), image)

    def find_exits(self, tracked_subjects, currently_red_ids, lost_ids, last_tracked_frames):

        for track_id in currently_red_ids:
            if len(np.where(tracked_subjects[:, -1] == track_id)[0]) == 0:
                currently_red_ids.remove(track_id)
                lost_ids.append(track_id)

        interest_subjects = [np.where(tracked_subjects[:, -1] == track_id)[0][0] for track_id in currently_red_ids if
                             len(np.where(tracked_subjects[:, -1] == track_id)[0]) == 1]
        if len(interest_subjects) == 0:
            return currently_red_ids, lost_ids

        yellow_indices = []
        for new_index, index in enumerate(interest_subjects):
            track_id = tracked_subjects[index, -1]
            for frame in reversed(range(len(last_tracked_frames))):
                try:
                    yellow_indices.append(
                        [frame, np.where(last_tracked_frames[frame][:, -1] == track_id)[0][0], new_index])
                    break
                except IndexError:
                    pass
        yellow_indices = np.array(yellow_indices)

        if len(yellow_indices) == 0:
            return currently_red_ids, lost_ids

        if self.track_point_position == "bottom":
            track_points = np.array([[int((subject[0] + subject[2]) / 2), subject[3], subject[-1]] for subject in
                                     tracked_subjects[interest_subjects, :]])
        elif self.track_point_position == "top":
            track_points = np.array([[int((subject[0] + subject[2]) / 2), subject[1], subject[-1]] for subject in
                                     tracked_subjects[interest_subjects, :]])
        else:
            raise ValueError
        track_points = track_points[yellow_indices[:, -1], :]

        green_indices = []
        for line_number in range(len(self.ms)):
            m = self.ms[line_number]
            b = self.bs[line_number]
            zone = self.zones[line_number]
            min_x = min(self.mouse_coordinates[line_number][0], self.mouse_coordinates[line_number + 1][0])
            max_x = max(self.mouse_coordinates[line_number][0], self.mouse_coordinates[line_number + 1][0])
            above_min = np.where(track_points[:, 0] >= min_x)
            below_max = np.where(track_points[:, 0] <= max_x)
            within_range = np.intersect1d(above_min, below_max)
            if zone == "up":
                focused = np.where(track_points[:, 0]*m + b >= track_points[:, 1])[0].tolist()
            else:
                focused = np.where(track_points[:, 0]*m + b <= track_points[:, 1])[0].tolist()
            green_indices += np.intersect1d(focused, within_range).tolist()
        green_indices = list(set(green_indices))

        for index in green_indices:
            prev_frame, prev_row, _ = yellow_indices[index]
            prev_pos = last_tracked_frames[prev_frame][prev_row, :]
            if self.track_point_position == "bottom":
                point_pos = [int((prev_pos[0] + prev_pos[2]) / 2), prev_pos[3]]
            elif self.track_point_position == "top":
                point_pos = [int((prev_pos[0] + prev_pos[2]) / 2), prev_pos[1]]
            else:
                raise ValueError
            for line_number in range(len(self.ms)):
                m = self.ms[line_number]
                b = self.bs[line_number]
                zone = self.zones[line_number]
                min_x = min(self.mouse_coordinates[line_number][0], self.mouse_coordinates[line_number + 1][0])
                max_x = max(self.mouse_coordinates[line_number][0], self.mouse_coordinates[line_number + 1][0])
                if min_x <= point_pos[0] <= max_x:
                    if zone == "up":
                        if point_pos[0]*m + b <= point_pos[1]:
                            try:
                                currently_red_ids.remove(track_points[index, -1])
                            except ValueError:
                                pass
                    else:
                        if point_pos[0] * m + b >= point_pos[1]:
                            try:
                                currently_red_ids.remove(track_points[index, -1])
                            except ValueError:
                                pass

        return currently_red_ids, lost_ids

    def detect_exits(self, tracked_subjects, currently_red_ids, lost_ids, last_tracked_frames):

        for track_id in currently_red_ids:
            if len(np.where(tracked_subjects[:, -1] == track_id)[0]) == 0:
                currently_red_ids.remove(track_id)
                lost_ids.append(track_id)

        interest_subjects = [np.where(tracked_subjects[:, -1] == track_id)[0][0] for track_id in currently_red_ids if
                             len(np.where(tracked_subjects[:, -1] == track_id)[0]) == 1]
        if len(interest_subjects) == 0:
            return currently_red_ids, lost_ids

        yellow_indices = []
        for new_index, index in enumerate(interest_subjects):
            track_id = tracked_subjects[index, -1]
            for frame in reversed(range(len(last_tracked_frames))):
                try:
                    yellow_indices.append(
                        [frame, np.where(last_tracked_frames[frame][:, -1] == track_id)[0][0], new_index])
                    break
                except IndexError:
                    pass
        if len(yellow_indices) == 0:
            return currently_red_ids, lost_ids

        yellow_indices = np.array(yellow_indices)
        if self.track_point_position == "bottom":
            track_points = np.array([[int((subject[0] + subject[2]) / 2), subject[3], subject[-1]] for subject in
                                     tracked_subjects[interest_subjects, :]])
        elif self.track_point_position == "top":
            track_points = np.array([[int((subject[0] + subject[2]) / 2), subject[1], subject[-1]] for subject in
                                     tracked_subjects[interest_subjects, :]])
        else:
            raise ValueError
        track_points = track_points[yellow_indices[:, -1], :]

        for index_num, index in enumerate(yellow_indices):
            x2, y2, track_id = track_points[index_num]
            prev_pos = last_tracked_frames[index[0]][index[1], :]
            if self.track_point_position == "bottom":
                x1, y1 = int((prev_pos[0] + prev_pos[2]) / 2), prev_pos[3]
            elif self.track_point_position == "top":
                x1, y1 = int((prev_pos[0] + prev_pos[2]) / 2), prev_pos[1]
            else:
                raise ValueError
            vertical_shift = y2 - y1
            horizontal_shift = x2 - x1
            if horizontal_shift == 0:
                suspect_m = 1e10
            elif vertical_shift == 0:
                suspect_m = 1e-10
            else:
                suspect_m = vertical_shift / horizontal_shift
            suspect_b = y2 - suspect_m*x2

            for line_number in range(len(self.ms)):
                m = self.ms[line_number]
                b = self.bs[line_number]
                zone = self.zones[line_number]
                variables = np.array([[-m, 1], [-suspect_m, 1]])  # -m*x + y =
                biases = np.array([b, suspect_b])  # b
                try:
                    intersection = np.linalg.solve(variables, biases)
                except np.linalg.LinAlgError:
                    continue
                min_x = min(self.mouse_coordinates[line_number][0], self.mouse_coordinates[line_number + 1][0])
                max_x = max(self.mouse_coordinates[line_number][0], self.mouse_coordinates[line_number + 1][0])
                min_y = min(self.mouse_coordinates[line_number][1], self.mouse_coordinates[line_number + 1][1])
                max_y = max(self.mouse_coordinates[line_number][1], self.mouse_coordinates[line_number + 1][1])
                if min_x <= intersection[0] <= max_x and min_y <= intersection[1] <= max_y:
                    if zone == "up":
                        if y1 >= m * x1 + b and y2 < m * x2 + b:
                            currently_red_ids.remove(track_id)
                            continue
                    elif zone == "down":
                        if y1 <= m * x1 + b and y2 > m * x2 + b:
                            currently_red_ids.remove(track_id)
                            continue
                    elif zone == "right":
                        if x1 >= (y1 - b) / m and x2 < (y2 - b) / m:
                            currently_red_ids.remove(track_id)
                            continue
                    elif zone == "left":
                        if x1 <= (y1 - b) / m and x2 > (y2 - b) / m:
                            currently_red_ids.remove(track_id)
                            continue

        return currently_red_ids, lost_ids

    def detect_inputs(self, tracked_subjects, last_tracked_frames, currently_red_ids, lost_ids):
        if self.track_point_position == "bottom":
            track_points = np.array([[int((subject[0] + subject[2]) / 2), subject[3]] for subject in tracked_subjects])
        elif self.track_point_position == "top":
            track_points = np.array([[int((subject[0] + subject[2]) / 2), subject[1]] for subject in tracked_subjects])
        else:
            raise ValueError

        yellow_indices = []
        for index, subject in enumerate(tracked_subjects):
            track_id = subject[-1]
            if track_id not in currently_red_ids:
                for frame in reversed(range(len(last_tracked_frames))):
                    try:
                        yellow_indices.append(
                            [frame, np.where(last_tracked_frames[frame][:, -1] == track_id)[0][0], index])
                        break
                    except IndexError:
                        pass

        red_indices = []
        for index in yellow_indices:
            x2, y2 = track_points[index[-1]]
            prev_pos = last_tracked_frames[index[0]][index[1], :]
            if self.track_point_position == "bottom":
                x1, y1 = int((prev_pos[0] + prev_pos[2]) / 2), prev_pos[3]
            elif self.track_point_position == "top":
                x1, y1 = int((prev_pos[0] + prev_pos[2]) / 2), prev_pos[1]
            else:
                raise ValueError
            vertical_shift = y2-y1
            horizontal_shift = x2-x1
            if horizontal_shift == 0:
                suspect_m = 1e10
            elif vertical_shift == 0:
                suspect_m = 1e-10
            else:
                suspect_m = vertical_shift/horizontal_shift
            suspect_b = y2 - suspect_m*x2

            for line_number in range(len(self.ms)):
                m = self.ms[line_number]
                b = self.bs[line_number]
                zone = self.zones[line_number]
                variables = np.array([[-m, 1], [-suspect_m, 1]])  # -m*x + y =
                biases = np.array([b, suspect_b]) # b
                try:
                    intersection = np.linalg.solve(variables, biases)
                except np.linalg.LinAlgError:
                    continue
                min_x = min(self.mouse_coordinates[line_number][0], self.mouse_coordinates[line_number+1][0])
                max_x = max(self.mouse_coordinates[line_number][0], self.mouse_coordinates[line_number+1][0])
                min_y = min(self.mouse_coordinates[line_number][1], self.mouse_coordinates[line_number + 1][1])
                max_y = max(self.mouse_coordinates[line_number][1], self.mouse_coordinates[line_number + 1][1])
                if min_x <= intersection[0] <= max_x and min_y <= intersection[1] <= max_y:
                    if zone == "up":
                        if y1 <= m*x1 + b and y2 > m*x2 + b:
                            red_indices.append(index[2])
                            continue
                    elif zone == "down":
                        if y1 >= m*x1 + b and y2 < m*x2 + b:
                            red_indices.append(index[2])
                            continue
                    elif zone == "right":
                        if x1 <= (y1-b)/m and x2 > (y2-b)/m:
                            red_indices.append(index[2])
                            continue
                    elif zone == "left":
                        if x1 >= (y1-b)/m and x2 < (y2-b)/m:
                            red_indices.append(index[2])
                            continue

        for track_id in lost_ids:
            location = np.where(tracked_subjects[:, -1] == track_id)[0]
            if len(location) == 1:
                red_indices.append(location[0])
                lost_ids.remove(track_id)

        return red_indices

    def draw_red_zone(self, image):
        thickness = int(self.im_height / 100)
        for start_index in range(len(self.mouse_coordinates) - 2):
            line_p1 = self.mouse_coordinates[start_index]
            line_p2 = self.mouse_coordinates[start_index + 1]
            image = cv2.line(image, line_p1, line_p2, (0, 0, 255), thickness)
        image = cv2.drawMarker(image, self.mouse_coordinates[-1], (0, 0, 255), cv2.MARKER_CROSS, 4 * thickness, thickness)
        return image

    def find_first_zone(self):
        self.zones = ["" for i in range(len(self.ms))]
        central_line = []
        for start_index in range(len(self.mouse_coordinates) - 2):
            x1, x2 = [pair[0] for pair in self.mouse_coordinates[start_index:start_index + 2]]
            min_x = min([x1, x2])
            max_x = max([x1, x2])
            if min_x <= self.mouse_coordinates[-1][0] <= max_x and abs(self.ms[start_index]) <= self.m_threshold:
                if self.mouse_coordinates[-1][1] >= self.mouse_coordinates[-1][0] * self.ms[start_index] + self.bs[
                    start_index]:
                    zone = "up"
                else:
                    zone = "down"
                central_line.append((start_index, x1, x2, zone))

        if len(central_line) == 0:
            for start_index in range(len(self.mouse_coordinates) - 2):
                x1, y1, x2, y2 = [value for pair in self.mouse_coordinates[start_index:start_index + 2] for value in
                                  pair]
                min_x = min([x1, x2])
                min_y = min([y1, y2])
                max_y = max([y1, y2])
                if min_y <= self.mouse_coordinates[-1][1] <= max_y and abs(self.ms[start_index]) > self.m_threshold:
                    if self.mouse_coordinates[-1][0] < min_x:
                        zone = "left"
                    else:
                        zone = "right"
                    central_line.append((start_index, x1, x2, zone))

        if len(central_line) == 0:
            return -1, -1, -1
        elif len(central_line) == 1:
            self.zones[central_line[0][0]] = central_line[0][-1]
            return central_line[0][0], central_line[0][1], central_line[0][2]
        else:
            if central_line[0][-1] == "up" or central_line[0][-1] == "down":
                ys = [self.ms[candidate[0]] * self.mouse_coordinates[-1][0] + self.bs[candidate[0]] for candidate in central_line]
                min_dis_up = 1e10
                index_up = -1
                min_dis_down = 1e10
                index_down = -1
                for index, y in enumerate(ys):
                    dis = self.mouse_coordinates[-1][1] - y
                    if dis < 0:
                        if abs(dis) < min_dis_up:
                            min_dis_up = abs(dis)
                            index_up = index
                    else:
                        if dis < min_dis_down:
                            min_dis_down = dis
                            index_down = index
                if index_up == -1:
                    self.zones[central_line[index_down][0]] = central_line[index_down][-1]
                    return central_line[index_down][0], central_line[index_down][1], central_line[index_down][2]
                elif index_down == -1:
                    self.zones[central_line[index_up][0]] = central_line[index_up][-1]
                    return central_line[index_up][0], central_line[index_up][1], central_line[index_up][2]
                else:
                    self.zones[central_line[index_up][0]] = central_line[index_up][-1]
                    return central_line[index_up][0], central_line[index_up][1], central_line[index_up][2]
            else:
                xs = [(self.mouse_coordinates[-1][1] - self.bs[candidate[0]])/self.ms[candidate[0]] for candidate in
                      central_line]
                min_dis_right = 1e10
                index_right = -1
                min_dis_left = 1e10
                index_left = -1
                for index, x in enumerate(xs):
                    dis = self.mouse_coordinates[-1][0] - x
                    if dis < 0:
                        if abs(dis) < min_dis_right:
                            min_dis_right = abs(dis)
                            index_right = index
                    else:
                        if dis < min_dis_left:
                            min_dis_left = dis
                            index_left = index
                if index_right == -1:
                    self.zones[central_line[index_left][0]] = central_line[index_left][-1]
                    return central_line[index_left][0], central_line[index_left][1], central_line[index_left][2]
                elif index_left == -1:
                    self.zones[central_line[index_right][0]] = central_line[index_right][-1]
                    return central_line[index_right][0], central_line[index_right][1], central_line[index_right][2]
                else:
                    self.zones[central_line[index_right][0]] = central_line[index_right][-1]
                    return central_line[index_right][0], central_line[index_right][1], central_line[index_right][2]

    def find_red_indices(self, tracked_subjects, last_tracked_frames, currently_red_ids, lost_ids):

        if self.track_point_position == "bottom":
            track_points = np.array([[int((subject[0] + subject[2]) / 2), subject[3]] for subject in tracked_subjects])
        elif self.track_point_position == "top":
            track_points = np.array([[int((subject[0] + subject[2]) / 2), subject[1]] for subject in tracked_subjects])
        else:
            raise ValueError

        yellow_indices = []
        for line_number in range(len(self.ms)):
            m = self.ms[line_number]
            b = self.bs[line_number]
            zone = self.zones[line_number]
            if zone == "up" or zone == "down":
                min_x = min(self.mouse_coordinates[line_number][0], self.mouse_coordinates[line_number + 1][0])
                max_x = max(self.mouse_coordinates[line_number][0], self.mouse_coordinates[line_number + 1][0])
                above_min = np.where(track_points[:, 0] >= min_x)
                below_max = np.where(track_points[:, 0] <= max_x)
                within_range = np.intersect1d(above_min, below_max)
                if zone == "up":
                    focused = np.where(track_points[:, 0] * m + b < track_points[:, 1])[0]
                else:
                    focused = np.where(track_points[:, 0] * m + b > track_points[:, 1])[0]
            else:
                min_y = min(self.mouse_coordinates[line_number][1], self.mouse_coordinates[line_number + 1][1])
                max_y = max(self.mouse_coordinates[line_number][1], self.mouse_coordinates[line_number + 1][1])
                above_min = np.where(track_points[:, 0] >= min_y)
                below_max = np.where(track_points[:, 0] <= max_y)
                within_range = np.intersect1d(above_min, below_max)
                if zone == "right":
                    focused = np.where((track_points[:, 1] - b)/m < track_points[:, 0])[0]
                else:
                    focused = np.where((track_points[:, 1] - b)/m > track_points[:, 0])[0]
            yellow_indices += np.intersect1d(focused, within_range).tolist()
        yellow_indices = list(set(yellow_indices))

        orange_indices = []
        for index in yellow_indices:
            track_id = tracked_subjects[index, -1]
            if track_id not in currently_red_ids:
                for frame in reversed(range(len(last_tracked_frames))):
                    try:
                        orange_indices.append(
                            [frame, np.where(last_tracked_frames[frame][:, -1] == track_id)[0][0], index])
                        break
                    except IndexError:
                        pass

        red_indices = []
        for index in orange_indices:
            prev_pos = last_tracked_frames[index[0]][index[1], :]
            if self.track_point_position == "bottom":
                point_pos = [int((prev_pos[0] + prev_pos[2]) / 2), prev_pos[3]]
            elif self.track_point_position == "top":
                point_pos = [int((prev_pos[0] + prev_pos[2]) / 2), prev_pos[1]]
            else:
                raise ValueError
            for line_number in range(len(self.ms)):
                m = self.ms[line_number]
                b = self.bs[line_number]
                zone = self.zones[line_number]
                if zone == "up" or zone == "down":
                    min_x = min(self.mouse_coordinates[line_number][0], self.mouse_coordinates[line_number + 1][0])
                    max_x = max(self.mouse_coordinates[line_number][0], self.mouse_coordinates[line_number + 1][0])
                    if min_x <= point_pos[0] <= max_x:
                        if zone == "up":
                            if point_pos[1] <= point_pos[0]*m + b:
                                red_indices.append(index[2])
                        else:
                            if point_pos[1] >= point_pos[0]*m + b:
                                red_indices.append(index[2])
                else:
                    min_y = min(self.mouse_coordinates[line_number][1], self.mouse_coordinates[line_number + 1][1])
                    max_y = max(self.mouse_coordinates[line_number][1], self.mouse_coordinates[line_number + 1][1])
                    if min_y <= point_pos[1] <= max_y:
                        if zone == "right":
                            if point_pos[0] <= (point_pos[1] - b)/m:
                                red_indices.append(index[2])
                        else:
                            if point_pos[0] >= (point_pos[1] - b)/m:
                                red_indices.append(index[2])
        for track_id in lost_ids:
            location = np.where(tracked_subjects[:, -1] == track_id)[0]
            if len(location) == 1:
                red_indices.append(location[0])
                lost_ids.remove(track_id)

        return list(set(red_indices))

    def read_line_from_text(self, path_to_text):
        if os.path.isfile(path_to_text):
            with open(path_to_text, 'r') as text_file:
                data = text_file.readlines()[0].strip()
            points = data.split(";")
            mouse_coordinates = [(int(point.split(",")[1]), int(point.split(",")[2])) for point in points]
        else:
            mouse_coordinates = []
        return mouse_coordinates

    def select_line(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_coordinates.append((x, y))

    def write_counts(self, img, currently_red_ids, all_red_ids):
        im_height = self.im_height
        display_height = self.display_height
        font_scale = 3 * im_height / display_height
        font_thickness = ceil(3 * im_height / display_height)
        t1 = "Overall count: %s" % len(all_red_ids)
        t2 = "Current count: %s" % len(currently_red_ids)
        t_size = cv2.getTextSize(t1, cv2.FONT_HERSHEY_PLAIN, font_scale, font_thickness)[0]
        cv2.putText(img, t1, (0, t_size[1]), cv2.FONT_HERSHEY_PLAIN, font_scale, [0, 0, 0],
                    font_thickness * 2)
        cv2.putText(img, t2, (0, ceil(2.1 * t_size[1])), cv2.FONT_HERSHEY_PLAIN, font_scale, [0, 0, 0],
                    font_thickness * 2)
        cv2.putText(img, t1, (0, t_size[1]), cv2.FONT_HERSHEY_PLAIN, font_scale, [255, 255, 255],
                    font_thickness)
        cv2.putText(img, t2, (0, ceil(2.1 * t_size[1])), cv2.FONT_HERSHEY_PLAIN, font_scale, [255, 255, 255],
                    font_thickness)
        return img

    def write_red_line_to_text(self, mouse_coordinates, text_file_path):
        string = []
        for index, point in enumerate(mouse_coordinates):
            string.append("point%s,%s,%s" % (index, mouse_coordinates[index][0], mouse_coordinates[index][1]))
        string = ";".join(string) + "\n"
        help = "# Three points as clicked in the GUI when this file doesn't exist.\nPoint number is followed by its x and y coordinates, defined as pixel position in the image.\nLast point is always the region of interest."
        with open(text_file_path, 'w') as text_file:
            text_file.writelines([string, help])

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
        last_n_tracked_frames = []
        all_detected = []
        currently_detected = []
        lost_detected = []
        while True:
            start = time.time()
            if not self.using_camera:
                grabbed, ori_im = self.vdo.read()
                if not grabbed:
                    break
                if not self.red_zone_defined:
                    self.define_red_zone(ori_im)
            else:
                if not self.red_zone_defined:
                    self.define_red_zone(self.stream)
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
                    # red_indices = self.find_red_indices(outputs, last_n_tracked_frames, currently_detected,
                    #                                     lost_detected)
                    red_indices = self.detect_inputs(outputs, last_n_tracked_frames, currently_detected,
                                                        lost_detected)
                    currently_detected += outputs[red_indices, -1].tolist()
                    currently_detected = list(set(currently_detected))
                    all_detected += currently_detected
                    all_detected = list(set(all_detected))

                    mass_centers = np.array([
                        [int((subject[0] + subject[2]) / 2), int((subject[1] + subject[3]) / 2)] for subject in outputs
                    ])
                    tl_corners = outputs[:, [0, 1]]
                    br_corners = outputs[:, [2, 3]]
                    tr_corners = np.concatenate((br_corners[:, 0][:, None], tl_corners[:, 1][:, None]), axis=1)
                    bl_corners = np.concatenate((tl_corners[:, 0][:, None], br_corners[:, 1][:, None]), axis=1)

                    currently_detected, lost_detected = self.detect_exits(outputs, currently_detected, lost_detected,
                                                                          last_n_tracked_frames)
                    red_indices = [np.where(outputs[:, -1] == track_id)[0][0] for track_id in currently_detected if
                                   len(np.where(outputs[:, -1] == track_id)[0]) == 1]
                    # mass_centers = mass_centers[red_indices]
                    # tl_corners = tl_corners[red_indices]
                    # br_corners = br_corners[red_indices]
                    # tr_corners = tr_corners[red_indices]
                    # bl_corners = bl_corners[red_indices]
                    analyzed_points_current_values = [mass_centers, tl_corners, tr_corners, bl_corners, br_corners]

                    # bbox_xyxy = outputs[:, :4][red_indices]
                    # identities = outputs[:, 4][red_indices]
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, 4]

                    for i in range(len(analyzed_points)):
                        frame_strs[i][0], frame_strs[i][1], frame_strs[i][2] = self.feat_to_str(
                            analyzed_points_current_values[i],
                            analyzed_points[i][0],
                            identities, analyzed_points[i][1],
                            frame_strs[i][0], frame_strs[i][1], frame_strs[i][2])

                    ori_im = draw_bboxes(self, ori_im, bbox_xyxy, identities, color=(0, 155, 0))
                    ori_im = draw_bboxes(self, ori_im, bbox_xyxy[red_indices], identities[red_indices], color="red")
                    if self.track_point_position == "bottom":
                        tracked_points = np.array(
                            [[int((subject[0] + subject[2]) / 2), subject[3]] for subject in outputs])
                    elif self.track_point_position == "top":
                        tracked_points = np.array(
                            [[int((subject[0] + subject[2]) / 2), subject[1]] for subject in outputs])

                    for point in tracked_points:
                        cv2.circle(ori_im, tuple(point), int(self.im_height / 100), (0, 0, 255), -1)

                    if len(last_n_tracked_frames) < self.n_frames:
                        last_n_tracked_frames.append(outputs)
                    else:
                        last_n_tracked_frames = last_n_tracked_frames[1:] + [outputs]
                else:
                    lost_detected += currently_detected
                    lost_detected = sorted(lost_detected)
                    currently_detected = []

            else:
                lost_detected += currently_detected
                lost_detected = sorted(lost_detected)
                currently_detected = []
                ori_im = self.draw_red_zone(ori_im)
            ori_im = self.write_counts(ori_im, currently_detected, all_detected)

            for i in range(len(frame_strs)):
                for j in range(len(frame_strs[i])):
                    self.outputs_dict[i][j].write("%s\n" % frame_strs[i][j])

            end = time.time()
            print("Source fps: {}, frame time: {:.3f}s, processing fps: {:.1f}, processed frames so far: {}".format(round(self.source_fps, 2), end - start, 1 / (end - start), real_frame), end='\r')

            if not bool(strtobool(self.args.ignore_display)):
                cv2.imshow("test", ori_im)
                # cv2.waitKey(1)

            if self.args.save_path is not None:
                self.output.write(ori_im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        print("Source fps: {}, frame time: {:.3f}s, processing fps: {:.1f}, processed frames so far: {}".format(round(self.source_fps, 2), end - start, 1 / (end - start), real_frame))
        if self.using_camera:
            self.stream.stop()
        self.close_text_files()


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--red_line_txt", type=str, default=None)
    parser.add_argument("--frame_rate", type=float, default=0)
    parser.add_argument("--track_point_position", type=str, default="top")
    parser.add_argument("--frames_memory_size", type=int, default=10)
    parser.add_argument("--yolo_cfg", type=str, default="YOLOv3/cfg/yolo_v3.cfg")
    parser.add_argument("--yolo_weights", type=str, default="YOLOv3/yolov3.weights")
    parser.add_argument("--yolo_names", type=str, default="YOLOv3/cfg/coco.names")
    parser.add_argument("--conf_thresh", type=float, default=0.5)
    parser.add_argument("--nms_thresh", type=float, default=0.4)
    parser.add_argument("--deepsort_checkpoint", type=str, default="deep_sort/deep/checkpoint/ckpt.t7")
    parser.add_argument("--max_dist", type=float, default=0.2)
    parser.add_argument("--max_age", type=int, default=70)
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
