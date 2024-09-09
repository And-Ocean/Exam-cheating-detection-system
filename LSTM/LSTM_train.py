import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt  # 引入 matplotlib
from ultralytics import YOLO
from LSTM_model import BiLSTMModel

# 检查是否有之前保存的模型文件
model_path = 'LSTM/my_bidirectional_lstm_model.pth'
continue_training = os.path.exists(model_path)

# 关键点坐标获取✓
model_yolo = YOLO("YOLO/yolov8n-pose.pt")
results = model_yolo(source="src\splitted\deliver_back_left1.mp4", 
            conf=0.5,
            iou=0.8,
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
sequence_length = 5
input_size = 34
current_sequence = []

for result in results:
    keypoint = result.keypoints[0].xyn.cpu().numpy()  # 形状为(1,17,2)
    if keypoint is not None:
        flattened_keypoint = keypoint.flatten()  # 每帧关键点的展平数据
        if len(flattened_keypoint) < input_size:  # 如果关键点数小于 input_size
            flattened_keypoint = np.pad(flattened_keypoint, (0, input_size - len(flattened_keypoint)), 'constant')  # 用0填充
        current_sequence.append(flattened_keypoint)  
        if len(current_sequence) == sequence_length:
            sequences.append(current_sequence)  
            current_sequence = []  # 重置以开始新的样本

# 确保每个序列的形状相同
sequences = [seq for seq in sequences if len(seq) == sequence_length]

# 转换为 numpy 数组
X = np.array(sequences, dtype=np.float32)

# 标签数据录入✓
labels = []
with open("src\splitted\deliver_back_left1.txt", 'r') as file:
    for line in file:
        line_labels = np.array(line.split(), dtype=np.int32)
        for i in range(0, len(line_labels), sequence_length):
            sample_label = line_labels[i:i + sequence_length]
            if len(sample_label) == sequence_length:
                labels.append(sample_label)

y = np.array(labels, dtype=np.float32)

# Tensor化
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# 创建数据加载器
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=5, shuffle=False, drop_last=True)

input_size = 34
hidden_size = 100
output_size = 1

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

# 存储每个 epoch 的损失和准确率
train_losses = []
accuracies = []

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)  
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    train_losses.append(avg_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    # 保存模型
    torch.save(model.state_dict(), model_path)
    print("Model saved to", model_path)

    # 评估模型
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            predicted = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
        accuracy = correct / total
        accuracies.append(accuracy)
        print(f'Accuracy: {accuracy:.4f}')

# 训练结束后可视化损失和准确率
epochs = range(1, num_epochs + 1)

plt.figure(figsize=(12, 6))

# 绘制损失曲线
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, 'b-', label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()

# 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(epochs, accuracies, 'g-', label='Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()

plt.tight_layout()
plt.show()
