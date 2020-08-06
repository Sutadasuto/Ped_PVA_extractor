# Ped_PVA_extractor

Repository based on https://github.com/ZQPei/deep_sort_pytorch for tracking pedestrians in videos or image sequences and calculating the instant position, velocity and acceleration on the x and y axes for the four points of the bounding boxes as well as their mass center.

The original implementation consists on two big parts: Multi-Object Detection using YOLOv3 and Multi-Object Tracking using Deep-SORT on the bounding boxes provided by YOLO. Our extension provides:
* saving the dynamic behavior of pedestrians (X and Y components of P/V/A) on text files
* converting those features in ready-for-ML numpy arrays (either for a single video or a folder containing a whole database)
* detecting people crossing an user-defined boundary in a specific direction (can be either a closed or open figure)
* counting people who have crossed such boundary.

## Pre-requisites
Within anaconda, you can set an environment from the environment.yml provided in this repository. Because of size, the parametes for YOLO and Deepsort are provided from a different source. To download them, inside the project folder you must:

1. Download YOLOv3 parameters:
```
cd YOLOv3/
wget https://pjreddie.com/media/files/yolov3.weights
cd ..
```
* Note that YOLOv3 is the default Object Detector. In the current repository version, Mask-RCNN using ResNet50 can be used instead of YOLO (when calling Mask-RCNN, the weights will be automatically downloaded by PyTorch), providing not only a bounding box but a mask of the detected pedestrians (masks are currently not being used but they are accesible by users).
* Ideally, any Object Detector can be used with the current repository, provided that:
  1. The Object Detector is a Python class
  2. The Class has an attribute called "class_names", which is a list of strings (where each element corresponds to one of all the possible labels that the detector can assign to an object)
  3. The Class has a __call__ function that accepts as argument an image (numpy array) and returns (as numpy arrays):
    * the bounding boxes of all (any class) the detected objects (in the format \[x_center, y_center, width, height])
    * the confidence(in the range \[0, 1]) of all the classifications
    * the label of the predicted class (as an integer in the range \[0-#number_of_class_names)) for each detected object
    * (Optional) The 2D masks of all the detected objects
  * Such additional Object Detector can be incorporated to the repository by adding it to the dictionary "self.detectors_dict" in the __init__ of the Tracker class in demo_yolo3_deepsort.py (look at the comments and use the incorporation of YOLO and Mask-RCNN as examples)

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
* ("--detector", type=str, default="yolov3") The Object Detector to use (currently "yolov3" and "mask_rcnn" are available)
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

## Stopping the program
If you kill the program, no text files nor videos will be saved in disk. To properly finish the program (even if there are still frames being processed), it is just enough to press the "q" key while on the window that is showing the frames being processed. The program will save the results processed until the last seen frame.

## Openvino support
This branch adds support for Intel's Openvino development. Current version works either with CPU or MYRIAD. However, additional setup is needed beyond the above mentioned pre-requisites. Particularly, you need to setup Openvino in your machine; you can follow a detailed tutorial (for Ubuntu) here: https://docs.openvinotoolkit.org/2020.4/openvino_docs_install_guides_installing_openvino_linux.html

**FOR FULL COMPATIBILITY** please follow the above mentioned tutorial after activating the conda environment provided in this repository (to allow Conda compatibility without interfering your system's python installation).

A sample using a person detector model and a person reidentification model (from the ones provided by Intel here: https://github.com/openvinotoolkit/open_model_zoo) is included here. To use them, provide "openvino" as --detector and --reidentifier arguments. For example:
```
python demo_yolo3_deepsort.py "path/to/video" --detector "openvino" --reidentifier "openvino" 
```
(Remember to set Openvino's environment variables before running any script using an Openvino model: https://docs.openvinotoolkit.org/2020.4/openvino_docs_install_guides_installing_openvino_linux.html#set-the-environment-variables)

Additionally, for using **MYRIAD** (Neural Compute Stick), you need to perform an additional setup in order to let your system interact with the USB stick: https://software.intel.com/content/www/us/en/develop/articles/get-started-with-neural-compute-stick.html

Once you system recognizes the MYRIAD device, you can mount the Openvino's models there by using the --use_movidius argument as True. For example:
```
python demo_yolo3_deepsort.py "path/to/video" --detector "openvino" --reidentifier "openvino" --use_movidius True
```

