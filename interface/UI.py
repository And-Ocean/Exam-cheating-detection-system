from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QLabel, QGridLayout
from producer import VideoProducer

class VideoPlayerApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi()

    def setupUi(self):
        self.address = ""
        self.setObjectName("MainWindow")
        self.resize(1280, 720)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        
        # Create grid layout
        gridLayout = QGridLayout(self.centralwidget)
        gridLayout.setContentsMargins(10, 10, 10, 10)
        gridLayout.setSpacing(10)

        # Create video label
        self.videoLabel = QLabel(self.centralwidget)
        self.videoLabel.setStyleSheet("background-color: rgb(0, 0, 0);")
        gridLayout.addWidget(self.videoLabel, 0, 1, 4, 1)  # Span 4 rows, 1 column

        # Initialize video producer
        self.videoProducer = VideoProducer(self.address)

        # Create buttons and other widgets
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.startVideo)
        gridLayout.addWidget(self.pushButton, 0, 0)
        gridLayout.setRowStretch(0, 2)  

        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.pauseVideo)
        gridLayout.addWidget(self.pushButton_2, 1, 0)
        gridLayout.setRowStretch(1, 2)  


        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setObjectName("pushButton_3")
        gridLayout.addWidget(self.pushButton_3, 2, 0)
        gridLayout.setRowStretch(2, 2)  


        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.lineEdit.setObjectName("lineEdit")
        gridLayout.addWidget(self.lineEdit, 4, 0)
        gridLayout.setRowStretch(4, 2)  


        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setObjectName("label_3")
        gridLayout.addWidget(self.label_3, 5, 0)
        gridLayout.setRowStretch(5, 2)  

        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label_4.setText("")
        self.label_4.setObjectName("label_4")
        gridLayout.addWidget(self.label_4, 6, 0)
        gridLayout.setRowStretch(6, 2)  

        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setObjectName("pushButton_4")
        gridLayout.addWidget(self.pushButton_4, 5, 0, 1, 1)
        
        # Set the central widget layout
        self.setCentralWidget(self.centralwidget)

        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 811, 26))
        self.menubar.setObjectName("menubar")
        self.setMenuBar(self.menubar)

        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)

        self.retranslateUi()

        self.pushButton_3.clicked.connect(self.lineEdit.clear)
        self.pushButton_4.clicked.connect(self.showLineEditText)

        QtCore.QMetaObject.connectSlotsByName(self)
        
        # Timer for updating video frame
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.labelUpdateTimer = QtCore.QTimer()
        self.labelUpdateTimer.timeout.connect(self.updateLabel4)
        self.labelUpdateTimer.start(1000)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "开始"))
        self.pushButton_2.setText(_translate("MainWindow", "暂停"))
        self.pushButton_3.setText(_translate("MainWindow", "切换"))
        self.lineEdit.setText(_translate("MainWindow", "请输入文件地址/流地址"))
        self.label_3.setText(_translate("MainWindow", "当前正在播放："))
        self.pushButton_4.setText(_translate("MainWindow", "确定"))

    def showLineEditText(self):
        self.address = self.lineEdit.text()
        if self.address == "0":
            self.videoProducer = VideoProducer(int(self.address))
        else:
            self.videoProducer = VideoProducer(self.address)
        
    def startVideo(self):
        self.timer.start(30)

    def pauseVideo(self):
        self.timer.stop()

    def update_frame(self):
        frame = self.videoProducer.get_frame()
        if frame is not None:
            height, width, channels = frame.shape
            bytes_per_line = channels * width
            qimage = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            self.videoLabel.setPixmap(QPixmap.fromImage(qimage))

    def updateLabel4(self):
        video_name = ""
        if self.address == "0":
            video_name = self.address
        else:
            video_name = self.videoProducer.get_video_name()
        self.label_4.setText(video_name)

    def launch(self):
        app = QtWidgets.QApplication(sys.argv)
        self.show()
        sys.exit(app.exec_())

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    UI = VideoPlayerApp()
    UI.launch()
    sys.exit(app.exec_())
