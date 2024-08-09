import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from ultralytics import YOLO
from sklearn.metrics import accuracy_score

# 定义YOLO模型并获取实时关键点数据
model_yolo = YOLO("yolov8n-pose.pt")

# 定义LSTM模型架构（与训练时相同）
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        h0 = torch.zeros(2 * 1, x.size(0), self.hidden_size).to(x.device)  # 2 for bidirection
        c0 = torch.zeros(2 * 1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.tanh(out)
        return out

input_size = 51
hidden_size = 100
output_size = 1
model_lstm = BiLSTMModel(input_size, hidden_size, output_size)

# 加载训练好的模型权重
model_lstm.load_state_dict(torch.load('E:/Exam-cheating-detection-system/Exam-cheating-detection-system/LSTM/my_bidirectional_lstm_model.pth'))

# 使用GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_lstm.to(device)

# 准备实时数据流
sequences = []
max_sequence_length = 100  # 假设一个合理的最大序列长度
stream = model_yolo(source=0, conf=0.5, show=True, save_txt=False, stream=True)

# 从YOLO模型获取实时关键点数据
for result in stream:
    keypoint = result.keypoints
    if keypoint is not None:

        keypoint = keypoint[0]  # 假设只有一个对象
        sequences.append(keypoint)
        if len(sequences) > max_sequence_length:
            sequences.pop(0)  # 保持序列长度不超过最大值

        # 当序列长度达到最大值时，进行预测
        if len(sequences) == max_sequence_length:
            sequence_tensor = torch.tensor(sequences, dtype=torch.float32).unsqueeze(0).to(device)  # 增加batch维度

            # 使用LSTM模型进行预测
            model_lstm.eval()
            with torch.no_grad():
                prediction = model_lstm(sequence_tensor).squeeze()
            
            # 打印预测结果
            predicted_label = (torch.sigmoid(prediction) > 0.5).float()
            if predicted_label.item() == 1:
                print("异常行为")
            else:
                print("正常行为")

            sequences = []  # 重置序列以开始新的检测周期

# # 读取真实标签（用于评估准确度）
# def load_true_labels(labels_file):
#     with open(labels_file, 'r') as file:
#         labels = [int(line.strip()) for line in file]
#     return labels

# labels_file = 'E:/video2/label/target2.txt'
# true_labels = load_true_labels(labels_file)
# true_labels = torch.tensor(true_labels, dtype=torch.float32).to(device)

# # 计算准确度（仅在测试数据上）
# def evaluate_model(true_labels, predictions):
#     accuracy = (true_labels == predictions).sum().item() / true_labels.size(0)
#     print(f'Accuracy: {accuracy:.4f}')

# # 计算预测准确度（假设有真实标签用于验证）
# predictions = []
# for result in stream:
#     keypoint = result.keypoints
#     if keypoint is not None:
#         keypoint = keypoint[0].cpu().numpy()
#         keypoints = keypoint[5:]
#         sequences.append(keypoints)
#         if len(sequences) > max_sequence_length:
#             sequences.pop(0)

#         if len(sequences) == max_sequence_length:
#             sequence_tensor = torch.tensor(sequences, dtype=torch.float32).unsqueeze(0).to(device)
#             model_lstm.eval()
#             with torch.no_grad():
#                 prediction = model_lstm(sequence_tensor).squeeze()
#             predicted_label = (torch.sigmoid(prediction) > 0.5).float()
#             predictions.append(predicted_label.item())
#             sequences = []

# predicted_labels = torch.tensor(predictions, dtype=torch.float32).to(device)
# evaluate_model(true_labels, predicted_labels)
