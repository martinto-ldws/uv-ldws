import sys
import cv2
import numpy as np
import yaml
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QGraphicsScene
from PyQt5.QtGui import QPixmap, QIcon, QImage
from PyQt5.QtCore import QTimer, Qt
from gui import Ui_MainWindow  # Import your GUI class from gui.py

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)  # Setup the UI defined in gui.py

        # Initialize variables
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.vision_enabled = True  # Controlled by toggleVisionIcon
        self.sound_enabled = True   # Controlled by toggleSoundIcon

        # Connect signals and slots
        self.picButton.clicked.connect(self.picFileDialog)
        self.vidButton.clicked.connect(self.vidFileDialog)
        self.visualBox.stateChanged.connect(self.toggleVisionIcon)
        self.soundBox.stateChanged.connect(self.toggleSoundIcon)
        self.applyButton.clicked.connect(self.applyConfig)

        # Initialize the graphics scene for graphicsView
        self.scene = QGraphicsScene()
        self.graphicsView.setScene(self.scene)

    # Provided functions
    def picFileDialog(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select a File", "", "Image files (*.jpg *.png)")
        if filename:
            path = Path(filename)
            self.picEdit.setText(str(path))

            # Load and process the image
            img = cv2.imread(str(path))
            if img is not None:
                if self.vision_enabled:
                    processed_img = self.lane_detection_pipeline(img)
                else:
                    processed_img = img
                self.display_image(processed_img)
            else:
                print("Failed to load image.")

    def vidFileDialog(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select a File", "", "Video files (*.mp4 *.mpeg *.avi)")
        if filename:
            path = Path(filename)
            self.vidEdit.setText(str(path))

            # Release previous video capture if any
            if self.cap is not None:
                self.cap.release()

            # Open the new video file
            self.cap = cv2.VideoCapture(str(path))

            # Start the timer to read frames
            self.timer.start(30)  # Adjust the interval as needed

    def toggleVisionIcon(self, state):
        if state == Qt.Checked:
            self.visualBox.setIcon(QIcon("Imágenes/visionON.svg"))
            self.vision_enabled = True
        else:
            self.visualBox.setIcon(QIcon("Imágenes/visionOFF.svg"))
            self.vision_enabled = False

    def toggleSoundIcon(self, state):
        if state == Qt.Checked:
            self.soundBox.setIcon(QIcon("Imágenes/soundON.svg"))
            self.sound_enabled = True
        else:
            self.soundBox.setIcon(QIcon("Imágenes/soundOFF.svg"))
            self.sound_enabled = False

    def applyConfig(self):
        file_name, _ = QFileDialog.getSaveFileName(self, 'Guardar archivo YAML', '', 'Archivos YAML (*.yaml)')
        if file_name:
            data = {
                'dataAcq': {
                    'dataSource': self.getDataSource(),
                    'picSource': self.picEdit.text(),
                    'vidSource': self.vidEdit.text(),
                },
                'alertConf': {
                    'visualAlert': 'ON' if self.visualBox.isChecked() else 'OFF',
                    'soundAlert': 'ON' if self.soundBox.isChecked() else 'OFF',
                },
                'senseConf': {
                    'visualSense': self.visualSenList.currentText(),
                    'soundSense': self.soundSenList.currentText(),
                },
                'camCalib': {
                    'focalLength': self.focalEdit.text(),
                    'imageCenterPoint': self.icpEdit.text(),
                    'efectivePixelSize': self.epsLine.text(),
                    'radialDistortion': {
                        'k1': self.k1Edit.text(),
                        'k2': self.k2Edit.text(),
                        'k3': self.k3Edit.text(),
                        'k4': self.k4Edit.text(),
                        'k5': self.k5Edit.text(),
                    },
                }
            }
            with open(file_name, 'w') as yaml_file:
                yaml.dump(data, yaml_file, default_flow_style=False)

    def getDataSource(self):
        if self.camButton.isChecked():
            return self.camButton.text()
        elif self.picButton.isChecked():
            return self.picButton.text()
        elif self.vidButton.isChecked():
            return self.vidButton.text()
        else:
            return "Unknown"

    # Lane detection pipeline functions
    def update_frame(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                if self.vision_enabled:
                    processed_frame = self.lane_detection_pipeline(frame)
                else:
                    processed_frame = frame
                self.display_image(processed_frame)
            else:
                self.cap.release()
                self.timer.stop()

    def display_image(self, img):
        # Convert the image to RGB (OpenCV uses BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channel = img.shape
        bytes_per_line = 3 * width

        # Create QImage from the image data
        q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Create QPixmap from QImage
        pixmap = QPixmap.fromImage(q_img)

        # Clear the previous content
        self.scene.clear()

        # Add the pixmap to the scene
        self.scene.addPixmap(pixmap)

        # Fit the pixmap in the view
        self.graphicsView.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def lane_detection_pipeline(self, frame):
        # Step 1: Apply thresholding
        binary = self.thresholding(frame)

        # Step 2: Perspective transform
        warped, Minv = self.perspective_transform(binary)

        # Step 3: Detect lane pixels and fit polynomial
        result = self.fit_polynomial(frame, warped, Minv)

        return result

    def thresholding(self, img):
        # Convert to HLS color space and separate the S channel
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        s_channel = hls[:, :, 2]

        # Apply Sobel operator in x direction
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        abs_sobelx = np.absolute(sobelx)
        if np.max(abs_sobelx) != 0:
            scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        else:
            scaled_sobel = np.uint8(255 * abs_sobelx)

        # Thresholds
        s_thresh = (170, 255)
        sx_thresh = (20, 100)

        # Apply thresholding
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

        sobel_binary = np.zeros_like(scaled_sobel)
        sobel_binary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

        # Combine the two binary thresholds
        combined_binary = np.zeros_like(sobel_binary)
        combined_binary[(s_binary == 1) | (sobel_binary == 1)] = 255

        return combined_binary

    def perspective_transform(self, img):
        img_size = (img.shape[1], img.shape[0])

        # Define source and destination points
        src = np.float32([
            [580, 460],
            [700, 460],
            [1040, 680],
            [260, 680]
        ])
        dst = np.float32([
            [260, 0],
            [1040, 0],
            [1040, 720],
            [260, 720]
        ])

        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)

        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

        return warped, Minv

    def fit_polynomial(self, original_img, binary_warped, Minv):
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)

        midpoint = int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        nwindows = 9
        margin = 100
        minpix = 50

        window_height = int(binary_warped.shape[0] / nwindows)

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height

            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin

            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        if len(leftx) == 0 or len(rightx) == 0:
            return original_img

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])

        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        left_pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        right_pts = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((left_pts, right_pts))

        cv2.fillPoly(color_warp, np.int32([pts]), (0, 255, 0))

        newwarp = cv2.warpPerspective(color_warp, Minv, (original_img.shape[1], original_img.shape[0]))
        result = cv2.addWeighted(original_img, 1, newwarp, 0.3, 0)

        return result

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
