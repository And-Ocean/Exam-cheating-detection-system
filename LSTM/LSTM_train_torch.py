import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split

from ultralytics import YOLO 
from torch import *

# model = YOLO("YOLO\yolov8n-pose.pt")

# results = model(source='E:/video1/video1.mp4', 
#             conf=0.5,
#             iou=0.6,
#             # 减少重复检测
#             half=True,
#             # 半精度推理，GPU加速
#             device=0,
#             stream_buffer=False,
#             # 在处理视频流时应该缓存所有帧
#             visualize=False,

#             show=True,
#             save=False,
#             save_frames=False,
#             save_txt=False,
#             save_conf=False,
#             show_labels=True,
#             show_conf=True,
#             show_boxes=True,
#             )


# 读取和解析数据
def load_data(data_dir):
    sequences = []
    labels = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, 'r') as file:
                lines = file.readlines()
                # sequence = []
                # for line in lines:
                #     values = list(map(float, line.strip().split()))
                #     keypoints = values[5:]  # 关键点坐标
                #     sequence.append(keypoints)
                # sequences.append(sequence)
                if 'abnormal' in filename:
                    labels.append(1)  # 异常行为
                else:
                    labels.append(0)  # 正常行为
    return sequences, labels

data_dir = 'E:/video1'
sequences, labels = load_data(data_dir)

# 序列填充
max_sequence_length = max(len(seq) for seq in sequences)
sequences_padded = np.zeros((len(sequences), max_sequence_length, 51), dtype=np.float32)
for i, seq in enumerate(sequences):
    for j, keypoints in enumerate(seq):
        sequences_padded[i, j, :] = keypoints

# 转换为numpy数组
X = np.array(sequences_padded)
y = np.array(labels)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为PyTorch张量
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 创建数据加载器
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义模型
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
model = BiLSTMModel(input_size, hidden_size, output_size)

# 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 评估模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        predicted = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')

# 保存模型
torch.save(model.state_dict(), 'LSTM/my_bidirectional_lstm_model.pth')
