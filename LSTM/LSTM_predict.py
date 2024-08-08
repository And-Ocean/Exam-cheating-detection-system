import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
data_dir = 'E:/video1.1'  
new_sequences, new_file_names = load_data_for_prediction(data_dir)

# 序列填充
max_sequence_length = max(len(seq) for seq in new_sequences)
new_sequences_padded = pad_sequences(new_sequences, maxlen=max_sequence_length, padding='post', dtype='float32')

# 转换为TensorFlow张量
X_new = tf.convert_to_tensor(new_sequences_padded, dtype=tf.float32)

# 加载训练好的模型
model = tf.keras.models.load_model('E:/Exam-cheating-detection-system/Exam-cheating-detection-system/LSTM/my_bidirectional_lstm_model') 

# 使用模型进行预测
predictions = model.predict(X_new)

# 解释预测结果
for filename, prediction in zip(new_file_names, predictions):
    if prediction >= 0.2:
        print(f'{filename}: 异常行为')
    else:
        print(f'{filename}: 正常行为')
