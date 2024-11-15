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
    def _init_(self):
        super(MainWindow, self)._init_()
        self.setupUi(self)  # Setup the UI defined in gui.py

        # Initialize variables
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.vision_enabled = True  # Controlled by toggleVisionIcon
        self.sound_enabled = True   # Controlled by toggleSoundIcon

        # Add a variable to enable/disable strict validation
        self.strict_validation = True  # Set to False to disable monitoring function

        # Initialize variables for lane detection
        self.left_fit_prev = None
        self.right_fit_prev = None

        # Threshold for maximum difference between frames (for temporal consistency)
        self.max_fit_diff = 0.001  # Adjust as needed

        # Connect signals and slots
        self.stopButton.clicked.connect(self.stop_video)
        self.startButton.clicked.connect(self.start_video)  # Start visualization
        self.cancelButton.clicked.connect(self.go_to_main_tab)  # Return to mainTab
        self.visualBox.stateChanged.connect(self.toggleVisionIcon)
        self.soundBox.stateChanged.connect(self.toggleSoundIcon)
        self.applyButton.clicked.connect(self.applyConfig)

        # Initialize the graphics scene for graphicsView
        self.scene = QGraphicsScene()
        self.graphicsView.setScene(self.scene)

        # Additional UI elements connections
        self.picButton.clicked.connect(self.picFileDialog)
        self.vidButton.clicked.connect(self.vidFileDialog)

    # Function to start video playback
    def start_video(self):
        data_source = self.getDataSource()
        if data_source == "Video":
            video_path = self.vidEdit.text()
            if video_path:
                # Release previous video capture if any
                if self.cap is not None:
                    self.cap.release()

                # Open the video file
                self.cap = cv2.VideoCapture(video_path)

                # Start the timer to read frames
                self.timer.start(30)  # Adjust the interval as needed
            else:
                print("No video file specified.")
        elif data_source == "Camara":
            # Release previous video capture if any
            if self.cap is not None:
                self.cap.release()

            # Open the camera (usually at index 0)
            self.cap = cv2.VideoCapture(0)

            # Start the timer to read frames
            self.timer.start(30)  # Adjust the interval as needed
        else:
            print("Data source not supported for start.")

    # Function to stop video playback
    def stop_video(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.timer.stop()
        # Clear the scene
        self.scene.clear()

    # Function to return to mainTab
    def go_to_main_tab(self):
        self.tabWidget.setCurrentIndex(0)  # Assuming mainTab is at index 0

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
                    if self.visualBox.isChecked():
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
                if self.visualBox.isChecked():
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
            # Step 1: Apply thresholding
            binary = self.thresholding(frame)

            # Step 2: Perspective transform
            warped, Minv = self.perspective_transform(binary)

            # Step 3: Detect lane pixels and fit polynomial
            result, detection_successful, left_fitx, right_fitx, ploty = self.fit_polynomial(frame, warped, Minv)

            # Step 4: Lane departure warning
            if detection_successful:
                result = self.calculate_lane_departure(result, left_fitx, right_fitx, ploty)

            return result
        except Exception as e:
            print(f"Error en lane_detection_pipeline: {e}")
            return frame

    def thresholding(self, img):
        try:
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
        except Exception as e:
            print(f"Error en thresholding: {e}")
            return np.zeros_like(img)

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

            warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

            return warped, Minv
        except Exception as e:
            print(f"Error en perspective_transform: {e}")
            return img, None

    def fit_polynomial(self, original_img, binary_warped, Minv):
        try:
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

            # Minimum number of pixels to consider a valid lane line
            min_lane_pixels = 500

            left_detected = len(leftx) >= min_lane_pixels
            right_detected = len(rightx) >= min_lane_pixels

            if not left_detected and not right_detected:
                detection_successful = False
                return original_img, detection_successful, None, None, None

            ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])

            left_fitx = None
            right_fitx = None

            if left_detected:
                left_fit = np.polyfit(lefty, leftx, 2)
                left_fitx = left_fit[0]ploty*2 + left_fit[1]*ploty + left_fit[2]
            else:
                self.left_fit_prev = None

            if right_detected:
                right_fit = np.polyfit(righty, rightx, 2)
                right_fitx = right_fit[0]ploty*2 + right_fit[1]*ploty + right_fit[2]
            else:
                self.right_fit_prev = None

            detection_successful = True

            # Call the monitoring function if strict_validation is enabled
            if self.strict_validation:
                detection_successful = self.is_valid_lane_detection(
                    left_fitx, right_fitx, ploty, binary_warped.shape[1], binary_warped.shape[0],
                    left_detected, right_detected)

            if not detection_successful:
                return original_img, detection_successful, left_fitx, right_fitx, ploty

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

            return result, detection_successful, left_fitx, right_fitx, ploty
        except Exception as e:
            print(f"Error en fit_polynomial: {e}")
            return original_img, False, None, None, None

    # Monitoring function to validate lane detection
    def is_valid_lane_detection(self, left_fitx, right_fitx, ploty, img_width, img_height,
                                left_detected, right_detected):
        try:
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
        except Exception as e:
            print(f"Error en is_valid_lane_detection: {e}")
            return False

    # Function to calculate lane departure and overlay warning
    def calculate_lane_departure(self, img, left_fitx, right_fitx, ploty):
        try:
            img_height, img_width = img.shape[:2]

            lane_departure_text = "Lane departure: none"

            # Define pixel-based departure threshold
            departure_threshold_pixels = 50  # Adjust this value for more flexibility

            if left_fitx is not None and right_fitx is not None:
                # Both lane lines detected
                lane_center = (left_fitx[-1] + right_fitx[-1]) / 2
                image_center = img_width / 2
                center_offset_pixels = image_center - lane_center

                if center_offset_pixels > departure_threshold_pixels:
                    lane_departure_text = "Lane departure: left"
                elif center_offset_pixels < -departure_threshold_pixels:
                    lane_departure_text = "Lane departure: right"
                else:
                    lane_departure_text = "Lane departure: none"

            elif left_fitx is not None:
                # Only left lane line detected
                left_lane_pos = left_fitx[-1]
                image_center = img_width / 2

                # Estimate the lane center assuming standard lane width in pixels
                estimated_lane_center = left_lane_pos + 700  # Approximate lane width in pixels
                center_offset_pixels = image_center - estimated_lane_center

                if center_offset_pixels > departure_threshold_pixels:
                    lane_departure_text = "Lane departure: left"
                elif center_offset_pixels < -departure_threshold_pixels:
                    lane_departure_text = "Lane departure: right"
                else:
                    lane_departure_text = "Lane departure: none"

            elif right_fitx is not None:
                # Only right lane line detected
                right_lane_pos = right_fitx[-1]
                image_center = img_width / 2

                # Estimate the lane center assuming standard lane width in pixels
                estimated_lane_center = right_lane_pos - 700  # Approximate lane width in pixels
                center_offset_pixels = image_center - estimated_lane_center

                if center_offset_pixels > departure_threshold_pixels:
                    lane_departure_text = "Lane departure: left"
                elif center_offset_pixels < -departure_threshold_pixels:
                    lane_departure_text = "Lane departure: right"
                else:
                    lane_departure_text = "Lane departure: none"
            else:
                # No lane lines detected
                lane_departure_text = "Lane departure: unknown"

            # Overlay the lane departure warning on the image
            cv2.putText(img, lane_departure_text, (50, img_height - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)

            return img
        except Exception as e:
            print(f"Error en calculate_lane_departure: {e}")
            return img

if _name_ == '_main_':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()

    # To disable the monitoring function for debugging, set strict_validation to False
    # mainWindow.strict_validation = False  # Uncomment this line to disable monitoring

    mainWindow.show()
    sys.exit(app.exec_())