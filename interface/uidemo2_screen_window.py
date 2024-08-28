import sys
import ctypes
import time
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton
from PyQt5.QtCore import QProcess


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Main Window with Embedded External Window")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.video_container = QWidget(self)
        self.video_container.setGeometry(10, 10, 640, 480)
        self.layout.addWidget(self.video_container)

        self.start_button = QPushButton("Start External Program", self)
        self.start_button.clicked.connect(self.start_external_program)
        self.layout.addWidget(self.start_button)

    def start_external_program(self):
        self.process = QProcess()
        self.process.start("这里填要启动的程序")  # 启动外部 Python 程序

        # 等待程序启动并创建窗口
        QTimer.singleShot(1000, self.embed_window)  # 1秒钟后尝试嵌入窗口

    def embed_window(self):
        hwnd = self.get_external_window_handle()
        if hwnd:
            ctypes.windll.user32.SetParent(hwnd, int(self.video_container.winId()))
            ctypes.windll.user32.ShowWindow(hwnd, 5)
            ctypes.windll.user32.SetWindowPos(hwnd, -1, 0, 0, 640, 480, 0x0040)

    def get_external_window_handle(self):
        # 使用窗口标题获取窗口句柄
        title = "里面填放视频那个窗口的标题"  # 替换为外部窗口的标题
        hwnd = ctypes.windll.user32.FindWindowW(None, title)#已知窗口类的话可以更换none，不然别动
        return hwnd


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())