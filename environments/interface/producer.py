import cv2

class VideoProducer:
    def __init__(self, video_source):
        self.video_source = video_source
        self.capture = cv2.VideoCapture(video_source)

    def get_frame(self):
        ret, frame = self.capture.read()
        if ret:
            return frame
        return None

    def release(self):
        self.capture.release()

    def get_video_name(self):
        return self.video_source.split('/')[-1]
