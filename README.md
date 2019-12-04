# Ped_PVA_extractor
Repository based on https://github.com/ZQPei/deep_sort_pytorch for tracking pedestrians in videos or image sequences and calculating the position, velocity and acceleration on the x and y axes for the four points of the bounding boxes as well as their mass center.

To run a sample demo on a single video, run: python demo_yolo3_deepsort.py "path/to/video" (you can provide a folder with a numbered sequence of images instead of a video file). The video with tracked subjects will be stored by default as demo.avi inside the project folder. The text files containing the extracted features will be stored by default in a folder called outputs inside the project folder.

To run the same analysis on a whole dataset, run: python preprocess_dataset.py "path/to/folder/containing/videos" (you can also provide a folder containing folders with numbered senquences of images). To save the videos with bounding boxes, add the next argument: --save_videos True. The resulting text files are ordered inside folders within a folder called outputs in the project folder by default. Aditionally, a numpy array (npy file) and a subject dictionary (csv) are stored in the same folder as a resume of all the tracked subjects in the database.
