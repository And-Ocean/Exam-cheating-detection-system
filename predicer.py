import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from ultralytics import YOLO
from sklearn.metrics import accuracy_score

# Initialize YOLO model
model_yolo = YOLO("yolov8n-pose.pt")

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
        out = out[:, -1, :]  # Get the output of the last time step
        out = self.fc(out)
        out = self.tanh(out)
        return out

class DetectAbnormalBehavior:
    def __init__(self, source):
        self.source = source

    def detect_abnormal_behavior(self):
        # Model parameters
        input_size = 34
        hidden_size = 100
        output_size = 1
        model_lstm = BiLSTMModel(input_size, hidden_size, output_size)

        # Load model weights
        model_lstm.load_state_dict(torch.load('bidirectional_lstm_model.pth'))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_lstm.to(device)

        # Initialize keypoint sequences for each detected person
        person_sequences = {}
        max_sequence_length = 30

        # Process the video stream using YOLO model
        stream = model_yolo(source=self.source,
                            conf=0.5,
                            iou=0.6,
                            half=True,
                            device=0,
                            stream_buffer=False,
                            visualize=False,
                            show=True,
                            save=False,
                            stream=True)


        for result in stream:
            keypoints = result.keypoints

            if len(keypoints) == 0:
                print("Warning: No person detected.")
                continue


            for i, keypoint in enumerate(keypoints):
                keypoint_np = keypoint.xyn.cpu().numpy()
                if keypoint_np is not None:
                    if i not in person_sequences:
                        person_sequences[i] = []
                    person_sequences[i].append(keypoint_np.flatten())


                    if len(person_sequences[i]) > max_sequence_length:
                        person_sequences[i].pop(0)


                    if len(person_sequences[i]) == max_sequence_length:
                        sequence_tensor = torch.tensor(person_sequences[i], dtype=torch.float32).unsqueeze(0).to(device)

                        model_lstm.eval()
                        with torch.no_grad():
                            prediction = model_lstm(sequence_tensor)

                            # Get model prediction
                            predicted_label = (torch.sigmoid(prediction) > 0.5).float().item()

                            # Clear the sequence after detection
                            person_sequences[i] = []

                            if predicted_label == 1:
                                return True
                            else:
                                return False

        return False
