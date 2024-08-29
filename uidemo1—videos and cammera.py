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
from predicer import DetectAbnormalBehavior
from PyQt5.QtCore import QThread, pyqtSignal


# Define a QThread subclass to run the detection
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


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        self.adress = ""
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(811, 604)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        # Create video label
        self.videoLabel = QLabel(self.centralwidget)
        self.videoLabel.setGeometry(QtCore.QRect(280, 10, 501, 531))
        self.videoLabel.setStyleSheet("background-color: rgb(0, 0, 0);")

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

        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
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

        self.pushButton_3.clicked.connect(self.lineEdit.clear)
        adress = self.pushButton_4.clicked.connect(self.showLineEditText)

        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # Timer for updating video frame
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "开始"))
        self.pushButton_2.setText(_translate("MainWindow", "暂停"))
        self.pushButton_3.setText(_translate("MainWindow", "切换"))
        self.lineEdit.setText(_translate("MainWindow", "请输入文件地址/流地址"))
        self.label_3.setText(_translate("MainWindow", "异常检测"))
        self.pushButton_4.setText(_translate("MainWindow", "确定"))

    def showLineEditText(self):
        self.adress = self.lineEdit.text()
        if self.adress == "0":
            self.videoProducer = VideoProducer(int(self.adress))
        else:
            self.videoProducer = VideoProducer(self.adress)

    def startVideo(self):
        # Initialize detect_predicter with the current address
        self.detectpredicter = DetectAbnormalBehavior(self.adress)

        # Start the video timer
        self.timer.start(90)  # Update every 30 ms

        self.detection_thread = DetectionThread(self.detectpredicter)
        self.detection_thread.update_label.connect(self.label_4.setText)
        self.detection_thread.start()
    def pauseVideo(self):
        self.timer.stop()
        if hasattr(self,'detection_thread'):
            self.detection_thread.terminate()

    def update_frame(self):
        # 获取视频帧
        frame = self.videoProducer.get_frame()

        if frame is not None:
            label_width = self.videoLabel.width()
            label_height = self.videoLabel.height()

            height, width, _ = frame.shape
            scale_x = label_width / width
            scale_y = label_height / height
            scale = min(scale_x, scale_y)
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized_frame = cv2.resize(frame, (new_width, new_height))
            qimage = QImage(resized_frame.data, new_width, new_height, new_width * 3, QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(qimage)
            self.videoLabel.setPixmap(pixmap)



class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
