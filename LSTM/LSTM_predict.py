import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from ultralytics import YOLO
from sklearn.metrics import accuracy_score
from LSTM_model import BiLSTMModel

# 初始化 YOLO 模型
model_yolo = YOLO("YOLO/yolov8n-pose.pt")

def detect_abnormal_behavior(source=0):
    """
    使用 YOLO 和 BiLSTM 检测视频流中的异常行为。
    
    参数:
        source: 视频源,默认为 0(摄像头)。
    
    返回值:
        bool: 如果检测到异常行为返回 True,否则返回 False。
    """
    # 模型参数
    input_size = 34  
    hidden_size = 100
    output_size = 1
    model_lstm = BiLSTMModel(input_size, hidden_size, output_size)

    # 加载模型权重
    model_lstm.load_state_dict(torch.load('LSTM/bidirectional_lstm_model.pth'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_lstm.to(device)
    
    # 初始化每个检测到的人的关键点序列
    person_sequences = {}
    max_sequence_length = 30

    # 使用 YOLO 模型处理视频流
    stream = model_yolo(source=source, 
                        conf=0.5,
                        iou=0.6,
                        half=True,
                        device=0,
                        stream_buffer=False,
                        visualize=False,
                        show=True,
                        save=False,
                        stream=True)

    # 遍历视频流中的每一帧
    for result in stream:
        keypoints = result.keypoints  # 检测到的人的关键点数据

        if len(keypoints) == 0:
            print("警告: 未检测考生。")
            continue

        # 检查当前帧中的每个检测到的人
        for i, keypoint in enumerate(keypoints):
            keypoint_np = keypoint.xyn.cpu().numpy()  # 获取每个人的关键点数据

            if keypoint_np is not None:
                if i not in person_sequences:
                    person_sequences[i] = []  # 初始化此人的序列

                # 展平关键点数据并加入对应人的序列
                person_sequences[i].append(keypoint_np.flatten())

                # 如果序列长度超过最大限制,则删除最早的帧
                if len(person_sequences[i]) > max_sequence_length:
                    person_sequences[i].pop(0)

                # 当序列达到最大长度时,进行行为检测
                if len(person_sequences[i]) == max_sequence_length:
                    sequence_tensor = torch.tensor(person_sequences[i], dtype=torch.float32).unsqueeze(0).to(device)

                    model_lstm.eval()
                    with torch.no_grad():
                        prediction = model_lstm(sequence_tensor)

                        # 输出模型预测结果
                        predicted_label = (torch.sigmoid(prediction) > 0.5).float().item()
                        
                        # 检测完后清空序列
                        person_sequences[i] = []

                        if predicted_label == 1:
                            return True  # 检测到异常行为返回 True
                        else:
                            return False  # 正常行为返回 False

    return False  # 如果没有检测到任何序列,返回 False
