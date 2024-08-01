from ultralytics import YOLO 

model = YOLO("E:\\Exam-cheating-detection-system\\Exam-cheating-detection-system\\ultralytics-main\\yolov8n-pose.pt")

results = model(source=0, show=True, conf=0.3, save_txt=True)
