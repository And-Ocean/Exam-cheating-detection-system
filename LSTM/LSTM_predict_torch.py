import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score

# 读取和解析数据函数
def load_data_for_prediction(data_dir):
    sequences = []
    file_names = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, 'r') as file:
                lines = file.readlines()
                sequence = []
                for line in lines:
                    values = list(map(float, line.strip().split()))
                    keypoints = values[5:]  # 关键点坐标
                    sequence.append(keypoints)
                sequences.append(sequence)
                file_names.append(filename)
    return sequences, file_names

# 加载新的数据
data_dir = 'E:/video2'
new_sequences, new_file_names = load_data_for_prediction(data_dir)

# 序列填充
new_sequences_padded = pad_sequence([torch.tensor(seq, dtype=torch.float32) for seq in new_sequences], batch_first=True, padding_value=0.0)

# 转换为PyTorch张量
X_new = new_sequences_padded

# 定义模型架构（与训练时相同）
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

# 加载训练好的模型权重
model.load_state_dict(torch.load('E:/Exam-cheating-detection-system/Exam-cheating-detection-system/LSTM/my_bidirectional_lstm_model.pth'))

# 使用GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 转换新数据为PyTorch张量
X_new = X_new.to(device)

# 使用模型进行预测
model.eval()
with torch.no_grad():
    predictions = model(X_new).squeeze()

# 加载真实标签
def load_true_labels(labels_file):
    with open(labels_file, 'r') as file:
        labels = [int(line.strip()) for line in file]
    return labels

labels_file = 'E:/video2/label/target2.txt'
true_labels = load_true_labels(labels_file)

true_labels = torch.tensor(true_labels, dtype=torch.float32).to(device)

# 计算准确度
predicted_labels = (torch.sigmoid(predictions) > 0.5).float()
accuracy = (predicted_labels == true_labels).sum().item() / true_labels.size(0)
print(f'Accuracy: {accuracy:.4f}')

# # 解释预测结果
# for filename, prediction in zip(new_file_names, predictions):
#     if torch.sigmoid(prediction) >= 0.5:
#         print(f'{filename}: 异常行为')
#     else:
#         print(f'{filename}: 正常行为')
