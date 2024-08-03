from ultralytics import YOLO 

model = YOLO("E:\\Exam-cheating-detection-system\\Exam-cheating-detection-system\\ultralytics-main\\yolov8n-pose.pt")

results = model(source='src\jpg\WIN_20240726_08_28_16_Pro.jpg', show=True, conf=0.3, save = True ,save_txt=True, save_conf = False)
