import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from ultralytics import YOLO
from sklearn.metrics import accuracy_score
import threading
import queue

class VideoStreamThread(threading.Thread):
    def __init__(self, video_source, queue):
        threading.Thread.__init__(self)
        self.video_source = video_source
        self.queue = queue
        self.model_yolo = YOLO("YOLO/yolov8n-pose.pt")
    
    def run(self):
        stream = self.model_yolo(source=self.video_source, 
                                 conf=0.5, 
                                 iou=0.6,
                                 device=0,
                                 stream_buffer=False, 
                                 show=False)
        for result in stream:
            keypoint = result.keypoints.xyn.cpu().numpy()
            if keypoint is not None:
                self.queue.put(keypoint.flatten())

class LSTMPredictionThread(threading.Thread):
    def __init__(self, model_lstm, sequence_queue, max_sequence_length, device):
        threading.Thread.__init__(self)
        self.model_lstm = model_lstm
        self.sequence_queue = sequence_queue
        self.max_sequence_length = max_sequence_length
        self.device = device
        self.sequences = []

    def run(self):
        while True:
            if not self.sequence_queue.empty():
                keypoint = self.sequence_queue.get()
                self.sequences.append(keypoint)
                if len(self.sequences) > self.max_sequence_length:
                    self.sequences.pop(0)

                if len(self.sequences) == self.max_sequence_length:
                    sequence_tensor = torch.tensor([self.sequences], dtype=torch.float32).to(self.device)
                    with torch.no_grad():
                        prediction = self.model_lstm(sequence_tensor).squeeze()
                        predicted_label = (torch.sigmoid(prediction) > 0.5).float()
                        if predicted_label.item() == 1:
                            print("异常行为检测到")
                        else:
                            print("行为正常")
                    self.sequences = []
