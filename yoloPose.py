from ultralytics import YOLO 
from torch import *

model = YOLO("yolov8n-pose.pt")

results = model(source="src\\mp4\\test4.mp4", 
            conf = 0.5,
            show=True, 
            save_txt=True,
            save_frames = True)
