# Ped_PVA_extractor
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

To run a sample demo on a single video, run: 
```
python demo_yolo3_deepsort.py "path/to/video"
```
You can provide a folder with a numbered sequence of images instead of a video file. Additionally, you can provide a device number instead of a path to grab video from a live camera (integrated webcams are typically device number 0, so you can replace "path/to/video" with "0" to test using your webcam). The video with tracked subjects will be stored by default as demo.avi inside the project folder. The text files containing the extracted features will be stored by default in a folder called outputs inside the project folder.

To run the same analysis on a whole dataset, run: 
```
python preprocess_dataset.py "path/to/folder/containing/videos"
```
You can also provide a folder containing folders with numbered senquences of images. To save the videos with bounding boxes, add the next argument: --save_videos True. The resulting text files are ordered inside folders within a folder called outputs in the project folder by default. Aditionally, a numpy array (npy file) and a subject dictionary (csv) are stored in the same folder as a resume of all the tracked subjects in the database.

Finally, you can run the tracker to spot subjects that crossed a user-defined line. To run on a single video, run:
```
python red_zone_tracker.py "path/to/video"
```
As with the first demo, you can also provide either a folder with images or a device number. When you run the demo, you will be shown a window with the first frame of the video (or the video stream, if using a camera); here you should select 3 points by clicking on the image: the first two points define a straight line, while the third one defines which side of the line is the region of interest. Once you select 3 points, the window will display the selected line as well as a circle indicating which is the region of interest given the straight line. Once you feel confortable with the selected points, press the "q" key to continue the program. You can reset the selected points at any moment pressing the "r" key. The outputs of this script are the same as the ones obtained with demo_yolo3_deepsort.py, but bounding box color will differ according to whether the person crossed the red line (red bounding box) or not (green bounding box); additionally, two counters are added to the screen: one to count how many people have crossed the line in the desired direction over all the video, and another one to count how many people currently tracked in the scene have crossed the line. To decide if a subject is inside the region of interest, their head must be inside such region.
