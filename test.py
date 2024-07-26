from ultralytics import YOLO 

model = YOLO("E:\\Exam-cheating-detection-system\\Exam-cheating-detection-system\\ultralytics-main\\yolov8n-pose.pt")

results = model(source="src\\mp4\\Desktop 2024.07.26 - 10.10.55.01.mp4", show=True, conf=0.3, save=False)
