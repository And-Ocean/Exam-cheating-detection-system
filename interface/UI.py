import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets,QtGui
from PyQt5.QtGui import QImage, QPixmap,QPainter,QPalette,QBrush
from PyQt5.QtCore import QTimer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from producer import VideoProducer
from PyQt5 import QtCore
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLCDNumber, QSlider,
                             QLabel,QMenuBar,QStatusBar, QPushButton, QFileDialog, QListWidget,QLineEdit)
from LSTM_predict import detect_abnormal_behavior

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
    def setupUi(self):
        #设置字体
        font1 = QtGui.QFont()
        font1.setPointSize(10)  # 括号里的数字可以设置成自己想要的字体大小
        font1.setFamily("Microsoft YaHei")  # 微软雅黑

        self.states = -1  # 视频播放状态，0表示暂停播放，1表示正在播放，-1表示播放结束
        self.camera_states=0 #摄像头使用状态，1表示使用摄像头
        self.resizeFlag=0 #窗口大小变化标志，窗口改变时赋值为1
        self.setWindowTitle("考试作弊检测系统")
        self.resize(1200, 800)

        palette = QPalette()
        palette.setBrush(QPalette.Background, QBrush(QPixmap("src/background.jpg")))  
        self.setPalette(palette)
        self.horizontalLayout = QHBoxLayout(self)
        # create buttons
        self.widget_left=QWidget(self)
        self.verticalLayout = QVBoxLayout(self.widget_left)
        self.pushButton_1 = QPushButton("选择视频",self.widget_left)
        self.pushButton_1.setFont(font1)
        self.pushButton_1.setFixedWidth(100)
        self.pushButton_1.setFixedHeight(40)
        self.verticalLayout.addWidget(self.pushButton_1)
        # self.pushButton.setGeometry(QtCore.QRect(50, 60, 141, 61))
        # self.pushButton.setObjectName("pushButton")
        self.pushButton_1.clicked.connect(self.chooseFile)

        self.pushButton_2 = QPushButton("实时视频",self.widget_left)
        self.pushButton_2.setFont(font1)
        self.pushButton_2.setFixedWidth(100)
        self.pushButton_2.setFixedHeight(40)
        self.verticalLayout.addWidget(self.pushButton_2)
        self.pushButton_2.clicked.connect(self.turn_on_camera)

        self.pushButton_3 = QPushButton("播放",self.widget_left)
        self.pushButton_3.setFont(font1)
        self.pushButton_3.setFixedWidth(100)
        self.pushButton_3.setFixedHeight(40)
        self.verticalLayout.addWidget(self.pushButton_3)
        self.pushButton_3.clicked.connect(self.startVideo)

        self.pushButton_4 = QPushButton("结束",self.widget_left)
        self.pushButton_4.setFont(font1)
        self.pushButton_4.setFixedWidth(100)
        self.pushButton_4.setFixedHeight(40)
        self.verticalLayout.addWidget(self.pushButton_4)
        self.pushButton_4.clicked.connect(self.stop)

        self.widget_right=QWidget(self)
        self.verticalLayout_2 = QVBoxLayout(self.widget_right)

        # Create video label
        self.videoLabel = QLabel(self.widget_right)
        self.videoLabel.setMinimumWidth(600)
        self.videoLabel.setMinimumHeight(400)
        self.videoLabel.setMaximumWidth(1500)
        self.videoLabel.setMaximumHeight(1000)
        self.verticalLayout_2.addWidget(self.videoLabel)
        # Initialize video producers
        self.videoProducer = VideoProducer('')

        self.label_1 = QLabel("当前正在播放：",self.widget_right)
        self.label_1.setFont(font1)
        self.label_1.setFixedWidth(450)
        self.label_1.setFixedHeight(40)
        self.verticalLayout_2.addWidget(self.label_1)

        self.label_3=QLabel("未检测到考生",self.widget_right)
        self.label_3.setFont(font1)
        self.label_3.setFixedWidth(400)
        self.label_3.setFixedHeight(40)
        self.verticalLayout_2.addWidget(self.label_3)
        
        #设置布局间距
        self.verticalLayout.setContentsMargins(0,0,0,100)
        #设置控件缩放比例
        self.verticalLayout_2.setStretch(0,10)
        self.verticalLayout_2.setStretch(1,1)
        self.verticalLayout_2.setStretch(2,1)

        self.horizontalLayout.addWidget(self.widget_left)
        self.horizontalLayout.addWidget(self.widget_right)
        self.horizontalLayout.setStretch(0,0)
        self.horizontalLayout.setStretch(1,6)


        QtCore.QMetaObject.connectSlotsByName(self)
        # Timer for updating video frame
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.labelUpadtetimer = QtCore.QTimer()
        self.labelUpadtetimer.timeout.connect(self.updateLabel_1)
        self.labelUpadtetimer.start(1000)
    def resizeEvent(self, event):
        self.resizeFlag=1
    def paintEvent(self,event):# set background_img
        painter = QPainter(self)
        painter.drawRect(self.rect())
        pixmap = QPixmap("src/background.jpg")
        painter.drawPixmap(self.rect(), pixmap)
    def chooseFile(self):
        url = QFileDialog.getOpenFileUrls()[0]
        file_url = [item.toLocalFile() for item in url][0]
        print("file url:",file_url)
        self.videoProducer=VideoProducer(file_url)
        self.timer.start(30)
        if self.states !=1:
            self.states = 1
            self.pushButton_3.setText('暂停')
    def turn_on_camera(self):
        if self.states!=1:
            self.videoProducer = VideoProducer(int("0"))
            self.timer.start(30)
            self.camera_states=1
            self.states=1
    def startVideo(self):
        if self.pushButton_3.text() == '暂停':
            self.timer.stop()
            self.pushButton_3.setText('播放')
            self.states=0
        else:
            self.timer.start(30)  # Update every 30 ms
            self.pushButton_3.setText('暂停')
            self.staets=1
    def stop(self):
        self.videoProducer = VideoProducer('') 
        self.camera_states=0
        self.states=-1
        self.updateLabel_1()
    def update_frame(self):
        # 获取视频帧
        frame = self.videoProducer.get_frame()
        flag=0
        if frame is not None: 
            height, width, _ = frame.shape
            if flag==0 or self.resizeFlag==1:  
                self.resizeFlag=0
                flag=1
                label_width = self.videoLabel.width()
                label_height = self.videoLabel.height()
            else:
                label_width,label_height=width,height
            scale_x = label_width / width
            scale_y = label_height / height
            scale = min(scale_x, scale_y)
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized_frame = cv2.resize(frame, (new_width, new_height))
            qimage = QImage(resized_frame.data, new_width, new_height, new_width * 3, QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(qimage)
            self.videoLabel.setPixmap(pixmap)
        frame = self.videoProducer.get_frame()
    def updateLabel_1(self):
        video_name = ""
        if self.camera_states==1:
            video_name ="本机摄像头"
        else:
            video_name =self.videoProducer.get_video_name()
        self.label_1.setText("当前正在播放："+video_name)
    def updateLabel_3(self, detect_state=False):
        if detect_state:
            self.label_3.setText("检测到作弊行为")
        else:
            self.label_3.setText("未发现异常")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    sys.exit(app.exec_())