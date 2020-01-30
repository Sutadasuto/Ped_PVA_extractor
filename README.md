# Ped_PVA_extractor

## Pre-requisites
Repository based on https://github.com/ZQPei/deep_sort_pytorch for tracking pedestrians in videos or image sequences and calculating the instant position, velocity and acceleration on the x and y axes for the four points of the bounding boxes as well as their mass center.

Within anaconda, you can set an environment from the environment.yml provided in this repository. Because of size, the parametes for YOLO and Deepsort are provided from a different source. To downloaded, insisde the project folder you must:

1. Download YOLOv3 parameters:
```
cd YOLOv3/
wget https://pjreddie.com/media/files/yolov3.weights
cd ..
```

2. Download deepsort parameters ckpt.t7:
```
cd deep_sort/deep/checkpoint
# download ckpt.t7 from 
https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6 to this folder
cd ../../../
```  

## Basic demo
To run a sample demo on a single video, run: 
```
python demo_yolo3_deepsort.py "path/to/video"
```
You can provide a folder with a numbered sequence of images instead of a video file. Additionally, you can provide a device number instead of a path to grab video from a live camera (integrated webcams are typically device number 0, so you can replace "path/to/video" with "0" to test using your webcam). The video with tracked subjects will be stored by default as demo.avi inside the project folder. The text files containing the extracted features will be stored by default in a folder called outputs inside the project folder.

Additional parameters:

* ("--output_dir", type=str, default=None) The folder where the resulting text files are saved
* ("--frame_rate", type=float, default=0) The desired frame rate to analyze (and write, if desired) the video (doesn't work for live stream)
* ("--conf_thresh", type=float, default=0.5) YOLO confidence threshold 
* ("--nms_thresh", type=float, default=0.4) YOLO Non-Maximal Suppression threshold
* ("--max_dist", type=float, default=0.2) Deepsort maximum difference distance for assigning existing IDs 
* ("--max_age", type=int, default=70) Deepsort maximum number of frames to store an ID before ignoring it
* ("--ignore_display", type=str, default="False") If ignore is chosen (True), during processing the program won't show the video being processed
* ("--display_width", type=int, default=800)
* ("--display_height", type=int, default=600)
* ("--save_path", type=str, default="demo.avi") The whole path to save the processed video. If None is provided, the resulting video won't be saved
* ("--use_cuda", type=str, default="True")

To run the same analysis on a whole dataset, run: 
```
python preprocess_dataset.py "path/to/folder/containing/videos"
```
You can also provide a folder containing folders with numbered senquences of images. To save the videos with bounding boxes, add the next argument: --save_videos True. The resulting text files are ordered inside folders within a folder called outputs in the project folder by default. Aditionally, a numpy array (npy file) and a subject dictionary (csv) are stored in the same folder as a resume of all the tracked subjects in the database.

## Red-line tracking
Finally, you can run the tracker to spot subjects that crossed a user-defined line. To run on a single video, run:
```
python red_zone_tracker.py "path/to/video"
```
As with the first demo, you can also provide either a folder with images or a device number. When you run the demo for the first time, you will be shown a window with the first frame of the video (or the video stream, if using a camera); here you should select at least 3 points by clicking on the image: the first two points define a straight line, while the third one defines which side of the line is the region of interest; whenever you make an additional click, that one becomes the new region of interest and all the previous ones create a continous line composed by line segments (each line segment is defined by two consecutive clicks).

Once you select at least 3 points, the window will display the selected line as well as a cross indicating which is the region of interest given the drawn line. Once you feel confortable with the selected points, press the "q" key to continue the program.

* a .txt file is generated in the source folder of the video with the same name as the video
* when analyzing a folder with images, the text file is saved in the folder containing the folder of interest
* when using an external device, the text file is saved in the project's root using the device number as file name
This text file saves the coordinates of the clicked points, so it is no longer necessary to select points when re-running the program. In the same location and with the same name, a JPG image is saved showing the red line in the image.

You can reset the selected points at any moment by pressing the "r" key. The outputs of this script are the same as the ones obtained with demo_yolo3_deepsort.py, but bounding box color will differ according to whether the person crossed the red line (red bounding box) or not (green bounding box); additionally, two counters are added to the screen: one to count how many people have crossed the line in the desired direction over all the video, and another one to count how many people currently tracked in the scene have crossed the line in the desired direction. Whenever a person crosses back the line, it is not longer counted and their bounding box turns green.

Additionally to the extra arguments that can be provided to demo_yolo3_deepsort.py, the following arguments can be used for red_zone_tracker.py:

* ("--red_line_txt", type=str, default=None) A path to a text file containing the desired points for the red line. If None, the program will look for it in the default path as explained in the last paragraph
* ("--track_point_position", type=str, default="top") Whether to use the approximate head (top) or feet (bottom) position to decide if a subject has crossed the line
* ("--frames_memory_size", type=int, default=10) How many frames in the past should be looked at to find the previous position of a person found in the current frame; if the person is not found in such frames, it is considered by default that the person didn't cross the line
