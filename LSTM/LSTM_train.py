import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# 读取和解析数据
def load_data(data_dir):
    sequences = []
    labels = []
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
                if 'abnormal' in filename:
                    labels.append(1)  # 异常行为
                else:
                    labels.append(0)  # 正常行为
    return sequences, labels

data_dir = 'runs\pose\predict2\labels'
sequences, labels = load_data(data_dir)


# 序列填充
max_sequence_length = max(len(seq) for seq in sequences)
sequences_padded = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', dtype='float32')

# 转换为numpy数组
X = np.array(sequences_padded)
y = np.array(labels)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 确保X和y为TensorFlow张量
X = tf.convert_to_tensor(X, dtype=tf.float32)
y = tf.convert_to_tensor(y, dtype=tf.float32)

# 构建LSTM模型
model = Sequential()
model.add(Masking(mask_value=0.0, input_shape=(max_sequence_length, 51)))  # 51是关键点坐标的维数（3*17）
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))

# 编译模型
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# 评估模型
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy}')

# 保存模型
model.save('LSTM/my_lstm_model')
