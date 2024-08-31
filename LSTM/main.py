from LSTM_predict import AbnormalBehaviorDetector

def main():
    # 设置 YOLO 和 LSTM 模型的路径
    yolo_model_path = "YOLO/yolov8n-pose.pt"
    lstm_model_path = "LSTM/bidirectional_lstm_model.pth"

    # 初始化异常行为检测器
    detector = AbnormalBehaviorDetector(yolo_model_path, lstm_model_path)

    try:
        # 开始检测，source=0 表示使用摄像头视频流
        detector.detect(source='src\splitted\deliver_right.mp4')
    except KeyboardInterrupt:
        # 捕捉用户中断（如 Ctrl+C）以优雅地退出程序
        print("检测中断，程序退出。")

if __name__ == "__main__":
    main()
