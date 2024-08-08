from ultralytics import YOLO 
from torch import *

model = YOLO("yolov8n-pose.pt")

results = model(source=0, 
            conf = 0.5,
            show=True, 
            # save_txt=True,
            save_txt=False,
            stream=True)


