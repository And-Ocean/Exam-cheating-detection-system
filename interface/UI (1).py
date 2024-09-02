import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets,QtGui
from PyQt5.QtGui import QImage, QPixmap,QPainter,QPalette,QBrush
from PyQt5.QtCore import QTimer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from predicer import DetectAbnormalBehavior
from PyQt5 import QtCore
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLCDNumber, QSlider,
                             QLabel,QMenuBar,QStatusBar, QPushButton, QFileDialog, QListWidget,QLineEdit)
from PyQt5.QtCore import QThread, pyqtSignal
from producer import VideoProducer
class DetectionThread(QThread):
    update_label = pyqtSignal(str)

    def __init__(self, detect_predicter, parent=None):
        super(DetectionThread, self).__init__(parent)
        self.detect_predicter = detect_predicter

    def run(self):
        result = self.detect_predicter.detect_abnormal_behavior()
        if result:
            self.update_label.emit("发现作弊")
        else:
            self.update_label.emit("没有异常")



class MainWindow(QWidget):
    def __init__(self,*args,**kwargs):
        super(MainWindow,self).__init__(*args,**kwargs)
        self.setupUi()
        self.paintEngine()
        self.show()
        self.setStyleSheet("""
QPushButton {
border: 2px solid #8f8f91;
border-radius: 6px;
background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,                
				stop: 0 #f6f7fa,                 
				stop: 1 #dadbde);
}
QPushButton:hover { 
background-color: #a1eeff; 
border-color: #00b3d9; 
}
QPushButton:pressed {
background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,                           
stop: 0 #d6f8ff,                               
stop: 1 #5de2ff);
}
QPushButton:flat {  
border: none; /* no border for a flat push button */
}
QPushButton:default {
border-color: navy; /* make the default button prominent */
}
QLabel{ 
border: 2px solid green;    
border-radius: 4px;
padding: 2px; 
}
        """)

    def createButton(self, text, slot):
        button = QPushButton(text, self.widget_left)
        button.setFont(self.font1)
        button.setFixedWidth(100)
        button.setFixedHeight(40)
        button.clicked.connect(slot)
        return button

    def setupUi(self):
        # 设置字体
        self.font1 = QtGui.QFont()
        self.font1.setPointSize(10)
        self.font1.setFamily("Microsoft YaHei")

        self.states = -1
        self.camera_states = 0
        self.resizeFlag = 0
        self.setWindowTitle("考试作弊检测系统")
        self.resize(1200, 800)

        # 设置背景
        palette = QPalette()
        palette.setBrush(QPalette.Background, QBrush(QPixmap("./background.jpg")))
        self.setPalette(palette)

        self.horizontalLayout = QHBoxLayout(self)

        # 左侧控件
        self.widget_left = QWidget(self)
        self.verticalLayout = QVBoxLayout(self.widget_left)

        self.pushButton_1 = self.createButton("选择视频", self.chooseFile)
        self.verticalLayout.addWidget(self.pushButton_1)
        self.pushButton_1.clicked.connect(lambda: self.label_3.setText(""))

        self.pushButton_2 = self.createButton("实时视频", self.turn_on_camera)
        self.verticalLayout.addWidget(self.pushButton_2)
        self.pushButton_2.clicked.connect(lambda: self.label_3.setText(""))

        self.pushButton_3 = self.createButton("开始检测（视频）", self.startVideo)
        self.verticalLayout.addWidget(self.pushButton_3)

        self.pushButton_4 = self.createButton("结束", self.stop)
        self.verticalLayout.addWidget(self.pushButton_4)

        # 右侧控件
        self.widget_right = QWidget(self)
        self.verticalLayout_2 = QVBoxLayout(self.widget_right)

        self.videoLabel = QLabel(self.widget_right)
        self.videoLabel.setMinimumWidth(600)
        self.videoLabel.setMinimumHeight(400)
        self.videoLabel.setMaximumWidth(1500)
        self.videoLabel.setMaximumHeight(1000)
        self.verticalLayout_2.addWidget(self.videoLabel)

        self.label_1 = QLabel("当前正在播放：", self.widget_right)
        self.label_1.setFont(self.font1)
        self.label_1.setFixedWidth(450)
        self.label_1.setFixedHeight(40)
        self.verticalLayout_2.addWidget(self.label_1)

        self.label_3 = QLabel("", self.widget_right)
        self.label_3.setFont(self.font1)
        self.label_3.setFixedWidth(400)
        self.label_3.setFixedHeight(40)
        self.verticalLayout_2.addWidget(self.label_3)

        # 设置布局间距
        self.verticalLayout.setContentsMargins(0, 0, 0, 100)
        self.verticalLayout_2.setStretch(0, 10)
        self.verticalLayout_2.setStretch(1, 1)
        self.verticalLayout_2.setStretch(2, 1)

        self.horizontalLayout.addWidget(self.widget_left)
        self.horizontalLayout.addWidget(self.widget_right)
        self.horizontalLayout.setStretch(0, 0)
        self.horizontalLayout.setStretch(1, 6)

        QtCore.QMetaObject.connectSlotsByName(self)

        # 定时器设置
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.labelUpdateTimer = QtCore.QTimer()
        self.labelUpdateTimer.timeout.connect(self.updateLabel_1)
        self.labelUpdateTimer.start(1000)
    def resizeEvent(self, event):
        self.resizeFlag=1
        print("Label resized to:", self.videoLabel.width(), self.videoLabel.height())

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawRect(self.rect())
        try:
            pixmap = QPixmap("./background.jpg")
            painter.drawPixmap(self.rect(), pixmap)
        except Exception as e:
            print("Error loading background image:", e)

    def chooseFile(self):
        try:

            url = QFileDialog.getOpenFileUrls()[0]
            if not url:
                return  # 用户未选择文件
            file_url = [item.toLocalFile() for item in url][0]
            print("file url:", file_url)
            self.videoProducer = VideoProducer(file_url)
            self.detectpredicter = DetectAbnormalBehavior(file_url)

            self.timer.start(30)

            self.detection_thread = DetectionThread(self.detectpredicter)
            self.detection_thread.update_label.connect(self.label_3.setText)
            self.detection_thread.start()
        except Exception as e:
            print("Error selecting file:", e)

    def turn_on_camera(self):
    
        if self.states != 1:
            if hasattr(self, 'detection_thread') and self.detection_thread.isRunning():
                return  # 如果线程正在运行，则不再启动新线程
            self.videoProducer = VideoProducer(0)
            self.detectpredicter = DetectAbnormalBehavior(0)
            self.timer.start(30)

            self.detection_thread = DetectionThread(self.detectpredicter)
            self.detection_thread.update_label.connect(self.label_3.setText)
            self.detection_thread.start()
    def startVideo(self):
        if self.pushButton_3.text() == '暂停':
            self.timer.stop()
            self.pushButton_3.setText('播放')
            self.states=0
        else:
            self.timer.start(30)
            self.pushButton_3.setText('暂停')
            self.states=1

    def stop(self):
        self.timer.stop()
        if hasattr(self, 'detection_thread'):
            self.detection_thread.quit()  # 请求线程退出
            self.detection_thread.wait()  # 等待线程安全退出

    def update_frame(self):
        frame = self.videoProducer.get_frame()
        if frame is not None:
            self.process_frame(frame)

    def process_frame(self, frame):
        height, width, _ = frame.shape
        label_width = self.videoLabel.width()
        label_height = self.videoLabel.height()

        scale_x = label_width / width
        scale_y = label_height / height
        scale = min(scale_x, scale_y)
        new_width = int(width * scale)
        new_height = int(height * scale)

        resized_frame = cv2.resize(frame, (new_width, new_height))
        qimage = QImage(resized_frame.data, new_width, new_height, new_width * 3, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qimage)
        self.videoLabel.setPixmap(pixmap)
    def updateLabel_1(self):
        video_name = ""
        if self.camera_states==1:
            video_name ="本机摄像头"
        else:
            video_name ="外部视频文件"
        self.label_1.setText("当前正在播放："+video_name)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    sys.exit(app.exec_())