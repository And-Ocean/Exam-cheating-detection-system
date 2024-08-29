import os
import sys

from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QLabel

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from ultralytics import YOLO
from sklearn.metrics import accuracy_score

from LSTM.LSTM_predict import detect_abnormal_behavior
from LSTM.LSTM_model import BiLSTMModel
from interface.UI import VideoPlayerApp

model_yolo = YOLO("YOLO/yolov8n-pose.pt")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    UI = VideoPlayerApp()
    UI.launch()
    sys.exit(app.exec_())
