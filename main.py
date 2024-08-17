import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from ultralytics import YOLO
from sklearn.metrics import accuracy_score

# 初始化 YOLO 模型
model_yolo = YOLO("YOLO/yolov8n-pose.pt")

class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device) 
        c0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        
        # 只取最后一个时间步的输出
        out = out[:, -1, :]  # 形状为 (batch_size, hidden_size * 2)
        out = self.fc(out)   # 形状为 (batch_size, output_size)
        out = self.tanh(out) # 形状为 (batch_size, output_size)
        return out


# 模型参数
input_size = 34  
hidden_size = 100
output_size = 1

# 加载 BiLSTM 模型
model_lstm = BiLSTMModel(input_size, hidden_size, output_size)
model_lstm.load_state_dict(torch.load('LSTM/my_bidirectional_lstm_model.pth'))

# 将模型放入 GPU (如果可用)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_lstm.to(device)

# 存储关键点序列
sequences = []
max_sequence_length = 19  

# 实时获取视频关键点
stream = model_yolo(source=0, 
            conf=0.5,
            iou=0.6,
            half=True,
            device=0,
            stream_buffer=False,
            visualize=False,
            show=True,
            save=False,
            stream=True
            )

# 处理每一帧视频
for result in stream:
    keypoint = result.keypoints.xyn.cpu().numpy()
    if keypoint is not None:
        sequences.append(keypoint.flatten())
        if len(sequences) > max_sequence_length:
            sequences.pop(0)

        if len(sequences) == max_sequence_length:
            sequence_tensor = torch.tensor(sequences, dtype=torch.float32).unsqueeze(0).to(device)

            model_lstm.eval()
            with torch.no_grad():
                prediction = model_lstm(sequence_tensor).squeeze()  # 输出形状应为 (1,)

            predicted_label = (torch.sigmoid(prediction) > 0.5).float()

            if predicted_label.item() == 1:
                print("异常行为检测到")
            else:
                print("行为正常")

            sequences = []  # 清空序列以处理下一个窗口
