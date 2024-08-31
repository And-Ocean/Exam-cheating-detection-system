import os
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from LSTM_model import BiLSTMModel

class AbnormalBehaviorDetector:
    def __init__(self, yolo_model_path, lstm_model_path, input_size=34, hidden_size=100, output_size=1, max_sequence_length=30, device=None):
        """
        初始化异常行为检测器。

        参数:
            yolo_model_path: YOLO 模型的路径。
            lstm_model_path: LSTM 模型的路径。
            input_size: LSTM 模型的输入大小。
            hidden_size: LSTM 模型的隐藏层大小。
            output_size: LSTM 模型的输出大小。
            max_sequence_length: 序列的最大长度。
            device: 计算设备 (CPU 或 GPU)。
        """
        self.model_yolo = YOLO(yolo_model_path)
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_lstm = BiLSTMModel(input_size, hidden_size, output_size)
        self.model_lstm.load_state_dict(torch.load(lstm_model_path))
        self.model_lstm.to(self.device)
        self.model_lstm.eval()
        self.max_sequence_length = max_sequence_length
        self.person_sequences = {}

    def detect(self, source=0):
        """
        使用 YOLO 和 BiLSTM 检测视频流中的异常行为。
        
        参数:
            source: 视频源,默认为 0(摄像头)。
        
        返回值:
            None: 该函数持续检测，输出检测结果，不会立即退出。
        """
        stream = self.model_yolo(source=source, 
                                 conf=0.5,
                                 iou=0.6,
                                 half=True,
                                 device=0,
                                 stream_buffer=False,
                                 visualize=False,
                                 show=True,
                                 save=False,
                                 stream=True)

        for result in stream:
            keypoints = result.keypoints

            if len(keypoints) == 0:
                print("警告: 未检测到考生。")
                continue

            for i, keypoint in enumerate(keypoints):
                keypoint_np = keypoint.xyn.cpu().numpy()

                if keypoint_np is not None:
                    if i not in self.person_sequences:
                        self.person_sequences[i] = []

                    self.person_sequences[i].append(keypoint_np.flatten())

                    if len(self.person_sequences[i]) > self.max_sequence_length:
                        self.person_sequences[i].pop(0)

                    if len(self.person_sequences[i]) == self.max_sequence_length:
                        sequence_tensor = torch.tensor(self.person_sequences[i], dtype=torch.float32).unsqueeze(0).to(self.device)

                        with torch.no_grad():
                            prediction = self.model_lstm(sequence_tensor)
                            predicted_label = (torch.sigmoid(prediction) > 0.5).float().item()

                            # 清空序列以继续检测
                            self.person_sequences[i] = []

                            if predicted_label == 1:
                                return i, True
                            else:
                                return i, False

        return i, False

