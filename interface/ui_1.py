from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QLabel
from producer import VideoProducer
from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        self.address = ""
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(811, 604)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        # Create video label
        self.videoLabel = QLabel(self.centralwidget)
        self.videoLabel.setGeometry(QtCore.QRect(280, 10, 640, 480))
        self.videoLabel.setStyleSheet("background-color: rgb(0, 0, 0);")

        # Initialize video producer
        self.videoProducer = VideoProducer(self.address)


        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(50, 60, 141, 61))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.startVideo)

        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(50, 140, 141, 61))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.pauseVideo)

        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(50, 220, 141, 61))
        self.pushButton_3.setObjectName("pushButton_3")

        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)  # Changed from QLabel to QLineEdit
        self.lineEdit.setGeometry(QtCore.QRect(10, 290, 251, 21))
        self.lineEdit.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.lineEdit.setObjectName("lineEdit")



        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(60, 410, 101, 20))
        self.label_3.setObjectName("label_3")

        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(20, 440, 191, 31))
        self.label_4.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label_4.setText("")
        self.label_4.setObjectName("label_4")

        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(50, 320, 141, 31))
        self.pushButton_4.setObjectName("pushButton_4")

        MainWindow.setCentralWidget(self.centralwidget)

        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 811, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        self.pushButton_3.clicked.connect(self.lineEdit.clear)  # Clears QLineEdit content
        address = self.pushButton_4.clicked.connect(self.showLineEditText)  # Connects to the new method

        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        # Timer for updating video frame
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.labelUpadtetimer = QtCore.QTimer()
        self.labelUpadtetimer.timeout.connect(self.updatelabel4)
        self.labelUpadtetimer.start(1000)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "开始"))
        self.pushButton_2.setText(_translate("MainWindow", "暂停"))
        self.pushButton_3.setText(_translate("MainWindow", "切换"))
        self.lineEdit.setText(_translate("MainWindow", "请输入文件地址/流地址"))  # Changed QLabel to QLineEdit
        self.label_3.setText(_translate("MainWindow", "当前正在播放："))
        self.pushButton_4.setText(_translate("MainWindow", "确定"))

    def showLineEditText(self):
        self.address = self.lineEdit.text()
        if self.address == "0":
            self.videoProducer = VideoProducer(int(self.address))
        else:
            self.videoProducer = VideoProducer(self.address)
         # Replace with desired handling
    def startVideo(self):
        self.timer.start(30)  # Update every 30 ms
    def pauseVideo(self):
        self.timer.stop()

    def update_frame(self):
        frame = self.videoProducer.get_frame()
        if frame is not None:
            # Convert the frame to QImage
            height, width, channels = frame.shape
            bytes_per_line = channels * width
            qimage = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            self.videoLabel.setPixmap(QPixmap.fromImage(qimage))
    def updatelabel4(self):
        video_name = ""
        if self.address== "0":
            video_name =self.address
        else:
            video_name =self.videoProducer.get_video_name()
        self.label_4.setText(video_name)


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())