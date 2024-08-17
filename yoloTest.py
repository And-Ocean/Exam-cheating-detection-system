from ultralytics import YOLO 

model = YOLO("YOLO\yolov8n-pose.pt")

results = model(source='src\jpg\WIN_20240726_08_28_16_Pro.jpg', conf=0.5, show=True, save_txt=False, stream=True)

for result in results:
    print(result.keypoints.data.dim())
    print(result.keypoints.data.size())
    