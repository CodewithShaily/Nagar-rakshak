import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, Qt  # QtCore imports
from PyQt5.QtGui import QImage  # Correct import for QImage
import cv2
import os
from collections import deque
from ultralytics import YOLO
import requests
import time

# Configurations
OUTPUT_DIR = "saved_frame"  # Same as weapon detection
SERVER_URL = "http://127.0.0.1:8000/api/images/"  # Same as weapon detection
CONFIDENCE_THRESHOLD = 0.5

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

class ViolenceDetection(QThread):
    changePixmap = pyqtSignal(QImage)  # Signal to update the UI with processed frames
    alertSent = pyqtSignal(str)  # Signal to notify that an alert was sent

    def __init__(self, token, location, receiver):
        super(ViolenceDetection, self).__init__()
        self.token = token
        self.location = location
        self.receiver = receiver
        self.running = False
        self.model = YOLO(r"weights\voilence_weights.pt")  # Replace with your model path
        self.frame_buffer = deque(maxlen=30)  # Buffer for processing frames
        self.mutex = QMutex()  # Mutex for thread safety

    def receive_frame(self, qt_image):
        """
        Receive frames emitted from the weapon detection model and process them.
        """
        self.mutex.lock()
        try:
            frame = self.convert_qt_image_to_cv(qt_image)
            self.frame_buffer.append(frame.copy())
        finally:
            self.mutex.unlock()

    def run(self):
        self.running = True

        while self.running:
            if len(self.frame_buffer) == 0:
                continue

            self.mutex.lock()
            frame = self.frame_buffer.popleft()
            self.mutex.unlock()

            # Run the YOLO model on the frame
            results = self.model(frame)
            for result in results:
                for cls in result.boxes.cls:
                    class_name = result.names[int(cls)]
                    if class_name == "violence":
                        print("Violence detected!")
                        self.save_and_alert(frame)
                        self.emit_frame_to_ui(frame)  # Emit the frame to UI
                        break

    def save_and_alert(self, frame):
        """
        Save detected frame and send an alert to the server.
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        frame_path = os.path.join(OUTPUT_DIR, f"violence_{timestamp}.jpg")
        cv2.imwrite(frame_path, frame)
        print(f"Saved detection frame: {frame_path}")

        # Send alert to the server
        try:
            headers = {"Authorization": f"Token {self.token}"}
            files = {"image": open(frame_path, "rb")}
            data = {"user_ID": self.token, "location": self.location, "alert_receiver": self.receiver}
            response = requests.post(SERVER_URL, files=files, headers=headers, data=data)
            if response.ok:
                print("Violence alert sent to the server.")
                self.alertSent.emit("violence")
            else:
                print("Failed to send violence alert.")
        except Exception as e:
            print(f"Error sending alert: {e}")

    def emit_frame_to_ui(self, frame):
        """
        Convert the frame to QImage and emit it to the UI.
        """
        rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bytesPerLine = frame.shape[1] * 3  # frame.shape[1] is width, 3 channels (RGB)
        convertToQtFormat = QImage(rgbImage.data, frame.shape[1], frame.shape[0], bytesPerLine, QImage.Format_RGB888)
        self.changePixmap.emit(convertToQtFormat)  # Emit the signal with the processed frame

    def stop(self):
        """
        Stop the thread safely.
        """
        self.running = False

    @staticmethod
    def convert_qt_image_to_cv(qt_image):
        """
        Convert QImage to OpenCV BGR image.
        """
        width = qt_image.width()
        height = qt_image.height()
        channels = 4 if qt_image.format() == QImage.Format_RGB32 else 3
        ptr = qt_image.bits()
        ptr.setsize(height * width * channels)
        array = np.array(ptr, dtype=np.uint8).reshape((height, width, channels))
        return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
