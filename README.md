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

To run a sample demo on a single video, run: python demo_yolo3_deepsort.py "path/to/video" (you can provide a folder with a numbered sequence of images instead of a video file). The video with tracked subjects will be stored by default as demo.avi inside the project folder. The text files containing the extracted features will be stored by default in a folder called outputs inside the project folder.

To run the same analysis on a whole dataset, run: python preprocess_dataset.py "path/to/folder/containing/videos" (you can also provide a folder containing folders with numbered senquences of images). To save the videos with bounding boxes, add the next argument: --save_videos True. The resulting text files are ordered inside folders within a folder called outputs in the project folder by default. Aditionally, a numpy array (npy file) and a subject dictionary (csv) are stored in the same folder as a resume of all the tracked subjects in the database.
