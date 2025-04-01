import sys
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QSlider, QHBoxLayout
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer

class VideoEditor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Editor with OpenCV")
        self.setGeometry(350, 100, 700, 500)
        self.init_ui()
        self.show()

    def init_ui(self):
        self.video_frame = QLabel()
        self.video_frame.setFixedSize(640, 360)
        self.video_frame.setStyleSheet("background-color: black;")

        openBtn = QPushButton('Open Video')
        openBtn.clicked.connect(self.open_file)

        self.playBtn = QPushButton('Play')
        self.playBtn.setEnabled(False)
        self.playBtn.clicked.connect(self.play_video)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.sliderMoved.connect(self.set_position)

        vbox = QVBoxLayout()
        vbox.addWidget(self.video_frame)
        vbox.addWidget(self.slider)

        hbox = QHBoxLayout()
        hbox.addWidget(openBtn)
        hbox.addWidget(self.playBtn)

        vbox.addLayout(hbox)
        self.setLayout(vbox)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame_slot)

    def open_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Video")

        if filename != '':
            self.cap = cv2.VideoCapture(filename)
            self.playBtn.setEnabled(True)
            self.slider.setEnabled(True)
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.slider.setMaximum(total_frames - 1)

    def play_video(self):
        if not self.timer.isActive():
            self.timer.start(30)
            self.playBtn.setText('Pause')
        else:
            self.timer.stop()
            self.playBtn.setText('Play')

    def next_frame_slot(self):
        ret, frame = self.cap.read()
        if ret:
            self.display_frame(frame)
            current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.slider.setValue(current_frame)
        else:
            self.timer.stop()
            self.cap.release()
            self.playBtn.setText('Play')

    def display_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.video_frame.width(), self.video_frame.height(), Qt.KeepAspectRatio)
        self.video_frame.setPixmap(QPixmap.fromImage(p))

    def set_position(self, position):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoEditor()
    sys.exit(app.exec_())
