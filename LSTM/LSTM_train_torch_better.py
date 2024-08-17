import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from ultralytics import YOLO 

# 检查是否有之前保存的模型文件
model_path = 'LSTM/my_bidirectional_lstm_model.pth'
continue_training = os.path.exists(model_path)

# 关键点坐标获取✓
model_yolo = YOLO("YOLO/yolov8n-pose.pt")
results = model_yolo(source='src/mp4/test4.mp4', 
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

# 关键点坐标数据录入✓
sequences = []
sequence_length = 19  
current_sequence = []

for result in results:
    keypoint = result.keypoints[0].xyn.cpu().numpy()  # 形状为(1,17,2)
    if keypoint is not None:
        current_sequence.append(keypoint.flatten())  # 每帧关键点的展平数据
        if len(current_sequence) == sequence_length:
            sequences.append(current_sequence)  
            current_sequence = []  # 重置以开始新的样本

# 转换为 numpy 数组
X = np.array(sequences, dtype=np.float32)

# 标签数据录入✓
labels = []
with open("src/label/test_label1.txt", 'r') as file:
    for line in file:
        line_labels = np.array(line.split(), dtype=np.int32)
        for i in range(0, len(line_labels), sequence_length):
            sample_label = line_labels[i:i + sequence_length]
            if len(sample_label) == sequence_length:
                labels.append(sample_label)

y = np.array(labels, dtype=np.float32)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tensor化
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 创建数据加载器
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True)

input_size = 34
hidden_size = 100
output_size = 1

# 定义模型
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
        out = self.fc(out)
        out = self.tanh(out)
        return out

model = BiLSTMModel(input_size, hidden_size, output_size)

# 如果存在已保存的模型，则加载它
if continue_training:
    model.load_state_dict(torch.load(model_path))
    print("Loaded existing model from", model_path)

# 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_epochs = 10
for epoch in torch.arange(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)  
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 保存模型
    torch.save(model.state_dict(), model_path)
    print("Model saved to", model_path)

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
