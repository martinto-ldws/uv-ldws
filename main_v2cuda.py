import sys
import cv2
import numpy as np
import yaml
import os
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

        # Add a variable to enable/disable strict validation
        self.strict_validation = True  # Set to False to disable monitoring function

        # Connect signals and slots
        self.stopButton.clicked.connect(self.stop_video)
        self.picBrowse.clicked.connect(self.picFileDialog)
        self.vidBrowse.clicked.connect(self.vidFileDialog)
        self.visualBox.stateChanged.connect(self.toggleVisionIcon)
        self.soundBox.stateChanged.connect(self.toggleSoundIcon)
        self.applyButton.clicked.connect(self.applyConfig)

        # Initialize the graphics scene for graphicsView
        self.scene = QGraphicsScene()
        self.graphicsView.setScene(self.scene)

    # Function to stop video playback
    def stop_video(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.timer.stop()
        # Clear the scene
        self.scene.clear()

    # Function to browse and load an image
    def picFileDialog(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Seleccionar Imagen", "", "Archivos de imagen (*.jpg *.png *.jpeg)")
        if filename:
            path = Path(filename)
            self.picEdit.setText(str(path))

            # Release any existing video capture
            if self.cap is not None:
                self.cap.release()
                self.cap = None
                self.timer.stop()

            # Load and process the image using np.fromfile and cv2.imdecode to handle special characters
            try:
                image_path = os.path.normpath(str(path))
                image_data = np.fromfile(image_path, dtype=np.uint8)
                img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                if img is not None:
                    if self.vision_enabled:
                        processed_img = self.lane_detection_pipeline(img)
                    else:
                        processed_img = img
                    self.display_image(processed_img)
                else:
                    print("Error al cargar la imagen.")
            except Exception as e:
                print(f"Error al cargar la imagen: {e}")

    # Function to browse and load a video
    def vidFileDialog(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Seleccionar Video", "", "Archivos de video (*.mp4 *.mpeg *.avi)")
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
            self.visualBox.setIcon(QIcon("imgss/visionON.svg"))
            self.vision_enabled = True
        else:
            self.visualBox.setIcon(QIcon("imgss/visionOFF.svg"))
            self.vision_enabled = False

    def toggleSoundIcon(self, state):
        if state == Qt.Checked:
            self.soundBox.setIcon(QIcon("imgss/soundON.svg"))
            self.sound_enabled = True
        else:
            self.soundBox.setIcon(QIcon("imgss/soundOFF.svg"))
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
                self.cap = None
                self.timer.stop()
        else:
            self.timer.stop()

    def display_image(self, img):
        try:
            # Convert the image to RGB (OpenCV uses BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, channel = img.shape
            bytes_per_line = 3 * width
            img = np.require(img, np.uint8, 'C')  # Ensure data is contiguous

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
        except Exception as e:
            print(f"Error en display_image: {e}")

    def lane_detection_pipeline(self, frame):
        try:
            # Upload the frame to GPU memory
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)

            # Step 1: Apply thresholding
            binary = self.thresholding(gpu_frame)

            # Step 2: Perspective transform
            warped, Minv = self.perspective_transform(binary)

            # Step 3: Detect lane pixels and fit polynomial
            result, detection_successful = self.fit_polynomial(frame, warped, Minv)

            if not detection_successful:
                # If detection failed, use the original frame
                result = frame.copy()

                # Optional: Overlay a warning message
                cv2.putText(result, 'Detección de carril fallida', (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            return result
        except Exception as e:
            print(f"Error en lane_detection_pipeline: {e}")
            return frame

    def thresholding(self, gpu_img):
        try:
            # Convert to HLS color space and separate the S channel
            gpu_hls = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2HLS)
            gpu_hls_channels = cv2.cuda.split(gpu_hls)
            gpu_s_channel = gpu_hls_channels[2]  # S channel

            # Apply Sobel operator in x direction
            gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)
            gpu_sobelx = cv2.cuda.Sobel(gpu_gray, cv2.CV_32F, 1, 0, ksize=3)
            gpu_abs_sobelx = cv2.cuda.abs(gpu_sobelx)

            # Normalize the Sobel result
            minVal, maxVal = cv2.cuda.minMaxLoc(gpu_abs_sobelx)[:2]
            if maxVal != 0:
                scale = 255 / maxVal
                gpu_scaled_sobel = cv2.cuda.multiply(gpu_abs_sobelx, scale)
                gpu_scaled_sobel = cv2.cuda.convertTo(gpu_scaled_sobel, cv2.CV_8U)
            else:
                gpu_scaled_sobel = cv2.cuda.convertTo(gpu_abs_sobelx, cv2.CV_8U)

            # Thresholds
            s_thresh = (170, 255)
            sx_thresh = (20, 100)

            # Apply thresholding on the S channel
            gpu_s_binary = cv2.cuda.threshold(gpu_s_channel, s_thresh[0], 255, cv2.THRESH_BINARY)[1]

            # Apply thresholding on the scaled Sobel result
            gpu_sobel_binary = cv2.cuda.threshold(gpu_scaled_sobel, sx_thresh[0], 255, cv2.THRESH_BINARY)[1]

            # Combine the two binary thresholds
            gpu_combined_binary = cv2.cuda.bitwise_or(gpu_s_binary, gpu_sobel_binary)

            # Download the result back to CPU memory
            combined_binary = gpu_combined_binary.download()

            return combined_binary
        except Exception as e:
            print(f"Error en thresholding: {e}")
            # If an error occurs, fall back to CPU processing
            return self.thresholding_cpu(gpu_img.download())

    def thresholding_cpu(self, img):
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
        try:
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

            # Use CUDA warpPerspective
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(img)

            gpu_warped = cv2.cuda.warpPerspective(gpu_img, M, img_size)
            warped = gpu_warped.download()

            return warped, Minv
        except Exception as e:
            print(f"Error en perspective_transform: {e}")
            # If an error occurs, fall back to CPU processing
            img_size = (img.shape[1], img.shape[0])
            warped = cv2.warpPerspective(img, M, img_size)
            return warped, Minv

    def fit_polynomial(self, original_img, binary_warped, Minv):
        # Histogram of the bottom half of the image
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

        min_lane_pixels = 500  # Adjust this threshold as needed

        left_detected = len(leftx) >= min_lane_pixels
        right_detected = len(rightx) >= min_lane_pixels

        if not left_detected and not right_detected:
            detection_successful = False
            return original_img, detection_successful

        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])

        left_fitx = None
        right_fitx = None

        if left_detected:
            left_fit = np.polyfit(lefty, leftx, 2)
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]

        if right_detected:
            right_fit = np.polyfit(righty, rightx, 2)
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        detection_successful = True

        # Call the monitoring function if strict_validation is enabled
        if self.strict_validation:
            detection_successful = self.is_valid_lane_detection(
                left_fitx, right_fitx, ploty, binary_warped.shape[1], binary_warped.shape[0],
                left_detected, right_detected)

        if not detection_successful:
            return original_img, detection_successful

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        if left_detected:
            left_line_pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            cv2.polylines(color_warp, np.int32([left_line_pts]), isClosed=False, color=(255, 0, 0), thickness=10)

        if right_detected:
            right_line_pts = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
            cv2.polylines(color_warp, np.int32([right_line_pts]), isClosed=False, color=(0, 0, 255), thickness=10)

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (original_img.shape[1], original_img.shape[0]))

        # Combine the result with the original image
        result = cv2.addWeighted(original_img, 1, newwarp, 0.3, 0)

        return result, detection_successful

    # Monitoring function to validate lane detection
    def is_valid_lane_detection(self, left_fitx, right_fitx, ploty, img_width, img_height, left_detected, right_detected):
        # Handle the cases when only one lane line is detected
        if left_detected and right_detected:
            lane_widths = right_fitx - left_fitx  # Width between lanes at all y positions

            # Check if lanes cross over
            if np.any(lane_widths <= 0):
                return False

            # Check if lane width is within acceptable range
            avg_lane_width = np.mean(lane_widths)

            # Define acceptable lane width range (in pixels)
            min_lane_width = 0.4 * img_width  # For example, 40% of image width
            max_lane_width = 0.8 * img_width  # For example, 80% of image width

            if not (min_lane_width < avg_lane_width < max_lane_width):
                return False

            # Check if lanes are centered within a margin
            lane_center = (left_fitx[-1] + right_fitx[-1]) / 2  # At the bottom of the image
            image_center = img_width / 2
            center_offset = abs(lane_center - image_center)

            # Define acceptable center offset (in pixels)
            max_center_offset = 0.1 * img_width  # For example, 10% of image width

            if center_offset > max_center_offset:
                return False

            # If all checks pass
            return True

        elif left_detected or right_detected:
            # If only one lane line is detected, we can accept it
            return True
        else:
            # No lane lines detected
            return False

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()

    # To disable the monitoring function for debugging, set strict_validation to False
    # mainWindow.strict_validation = False  # Uncomment this line to disable monitoring

    mainWindow.show()
    sys.exit(app.exec_())
