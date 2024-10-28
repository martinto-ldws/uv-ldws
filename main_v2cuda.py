import sys
import cv2
import numpy as np
import yaml
import os
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QGraphicsScene
from PyQt5.QtGui import QPixmap, QIcon, QImage
from PyQt5.QtCore import QTimer, Qt
from gui import Ui_MainWindow  # Importa tu clase GUI desde gui.py

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)  # Configura la interfaz definida en gui.py

        # Inicializar variables
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.vision_enabled = True  # Controlado por toggleVisionIcon
        self.sound_enabled = True   # Controlado por toggleSoundIcon

        # Añadir una variable para habilitar/deshabilitar la validación estricta
        self.strict_validation = True  # Establecer en False para deshabilitar la función de supervisión

        # Inicializar variables para la detección de carriles
        self.left_fit_prev = None
        self.right_fit_prev = None

        # Umbral para la diferencia máxima entre cuadros (para consistencia temporal)
        self.max_fit_diff = 0.001  # Ajustar según sea necesario

        # Conectar señales y slots
        self.stopButton.clicked.connect(self.stop_video)
        self.visualBox.stateChanged.connect(self.toggleVisionIcon)
        self.soundBox.stateChanged.connect(self.toggleSoundIcon)
        self.applyButton.clicked.connect(self.applyConfig)

        # Inicializar la escena gráfica para graphicsView
        self.scene = QGraphicsScene()
        self.graphicsView.setScene(self.scene)

    # Función para detener la reproducción de video
    def stop_video(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.timer.stop()
        # Limpiar la escena
        self.scene.clear()

    # Función para explorar y cargar una imagen
    def picFileDialog(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Seleccionar Imagen", "", "Archivos de imagen (*.jpg *.png *.jpeg)")
        if filename:
            path = Path(filename)
            self.picEdit.setText(str(path))

            # Liberar cualquier captura de video existente
            if self.cap is not None:
                self.cap.release()
                self.cap = None
                self.timer.stop()

            # Cargar y procesar la imagen
            try:
                img = cv2.imread(str(path))
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

    # Función para explorar y cargar un video
    def vidFileDialog(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Seleccionar Video", "", "Archivos de video (*.mp4 *.mpeg *.avi)")
        if filename:
            path = Path(filename)
            self.vidEdit.setText(str(path))

            # Liberar captura de video anterior si existe
            if self.cap is not None:
                self.cap.release()

            # Abrir el nuevo archivo de video
            self.cap = cv2.VideoCapture(str(path))

            # Iniciar el temporizador para leer fotogramas
            self.timer.start(30)  # Ajusta el intervalo según sea necesario

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

    # Funciones del pipeline de detección de carriles
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
            # Convertir la imagen a RGB (OpenCV utiliza BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, channel = img.shape
            bytes_per_line = 3 * width
            img = np.require(img, np.uint8, 'C')  # Asegurar que los datos sean contiguos

            # Crear QImage a partir de los datos de la imagen
            q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)

            # Crear QPixmap a partir de QImage
            pixmap = QPixmap.fromImage(q_img)

            # Limpiar el contenido previo
            self.scene.clear()

            # Agregar el pixmap a la escena
            self.scene.addPixmap(pixmap)

            # Ajustar el pixmap en la vista
            self.graphicsView.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        except Exception as e:
            print(f"Error en display_image: {e}")

    def lane_detection_pipeline(self, frame):
        try:
            # Subir el frame a la GPU
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)

            # Paso 1: Aplicar umbralización
            gpu_binary = self.thresholding(gpu_frame)

            # Paso 2: Transformación de perspectiva
            gpu_warped, Minv = self.perspective_transform(gpu_binary)

            # Paso 3: Detectar píxeles de carril y ajustar polinomios
            # Descargamos la imagen procesada a CPU para usar funciones de CPU en fit_polynomial
            warped = gpu_warped.download()
            result, detection_successful = self.fit_polynomial(frame, warped, Minv)

            if not detection_successful:
                # Si la detección falló, usar el frame original
                result = frame.copy()

                # Opcional: Superponer un mensaje de advertencia
                cv2.putText(result, 'Detección de carril fallida', (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            return result
        except Exception as e:
            print(f"Error en lane_detection_pipeline: {e}")
            return frame

    def thresholding(self, gpu_img):
        try:
            # Convertir a espacio de color HLS
            gpu_hls = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2HLS)

            # Separar los canales H, L y S
            gpu_hls_channels = cv2.cuda_GpuMat()
            gpu_hls_channels.create(gpu_hls.size(), gpu_hls.type())
            gpu_hls_channels = cv2.cuda.split(gpu_hls)

            gpu_s_channel = gpu_hls_channels[2]

            # Aplicar operador Sobel en dirección x
            gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)
            gpu_sobelx = cv2.cuda.Sobel(gpu_gray, cv2.CV_32F, 1, 0, ksize=3)

            gpu_abs_sobelx = cv2.cuda.abs(gpu_sobelx)

            # Escalar la magnitud de Sobel a rango [0, 255]
            min_val, max_val = cv2.cuda.minMaxLoc(gpu_abs_sobelx)[:2]
            if max_val != 0:
                scale_factor = 255.0 / max_val
            else:
                scale_factor = 0
            gpu_scaled_sobel = cv2.cuda.multiply(gpu_abs_sobelx, scale_factor)

            # Umbrales
            s_thresh_min = 170
            s_thresh_max = 255
            sx_thresh_min = 20
            sx_thresh_max = 100

            # Aplicar umbralización en el canal S
            gpu_s_binary = cv2.cuda.threshold(gpu_s_channel, s_thresh_min, 255, cv2.THRESH_BINARY)[1]

            # Aplicar umbralización en la magnitud de Sobel
            gpu_sobel_binary = cv2.cuda.threshold(gpu_scaled_sobel, sx_thresh_min, 255, cv2.THRESH_BINARY)[1]

            # Combinar las dos umbralizaciones binarias
            gpu_combined_binary = cv2.cuda.bitwise_or(gpu_s_binary, gpu_sobel_binary)

            return gpu_combined_binary

        except Exception as e:
            print(f"Error en thresholding: {e}")
            # Si hay un error, usar la versión de CPU como respaldo
            return self.thresholding_cpu(gpu_img.download())

    def thresholding_cpu(self, img):
        try:
            # Convertir al espacio de color HLS y separar el canal S
            hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
            s_channel = hls[:, :, 2]

            # Aplicar operador Sobel en dirección x
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
            abs_sobelx = np.absolute(sobelx)
            if np.max(abs_sobelx) != 0:
                scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
            else:
                scaled_sobel = np.uint8(255 * abs_sobelx)

            # Umbrales
            s_thresh = (170, 255)
            sx_thresh = (20, 100)

            # Aplicar umbralización
            s_binary = np.zeros_like(s_channel)
            s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 255

            sobel_binary = np.zeros_like(scaled_sobel)
            sobel_binary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 255

            # Combinar las dos umbralizaciones binarias
            combined_binary = cv2.bitwise_or(s_binary, sobel_binary)

            return combined_binary
        except Exception as e:
            print(f"Error en thresholding_cpu: {e}")
            return None

    def perspective_transform(self, gpu_img):
        try:
            img_size = (gpu_img.size()[1], gpu_img.size()[0])  # (ancho, alto)

            # Definir puntos de origen y destino
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

            # Calcular la matriz de transformación de perspectiva
            M = cv2.getPerspectiveTransform(src, dst)
            Minv = cv2.getPerspectiveTransform(dst, src)

            # Aplicar la transformación de perspectiva
            gpu_warped = cv2.cuda.warpPerspective(gpu_img, M, img_size)

            return gpu_warped, Minv
        except Exception as e:
            print(f"Error en perspective_transform: {e}")
            # Si hay un error, usar la versión de CPU como respaldo
            warped = cv2.warpPerspective(gpu_img.download(), M, img_size)
            return cv2.cuda_GpuMat().upload(warped), Minv

    def fit_polynomial(self, original_img, binary_warped, Minv):
        try:
            # Convertir imagen binaria a formato adecuado
            binary_warped = binary_warped.astype(np.uint8)

            # Histograma del half inferior de la imagen
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

            # Umbrales ajustados para el número mínimo y máximo de píxeles
            min_lane_pixels = 500   # Número mínimo de píxeles para considerar una línea de carril válida
            max_lane_pixels = 15000 # Número máximo de píxeles para descartar líneas de carril irreales

            left_detected = min_lane_pixels <= len(leftx) <= max_lane_pixels
            right_detected = min_lane_pixels <= len(rightx) <= max_lane_pixels

            if not left_detected and not right_detected:
                detection_successful = False
                return original_img, detection_successful

            ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])

            left_fitx = None
            right_fitx = None

            if left_detected:
                left_fit = np.polyfit(lefty, leftx, 2)
                left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            else:
                self.left_fit_prev = None

            if right_detected:
                right_fit = np.polyfit(righty, rightx, 2)
                right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            else:
                self.right_fit_prev = None

            detection_successful = True

            # Llamar a la función de supervisión si la validación estricta está habilitada
            if self.strict_validation:
                detection_successful = self.is_valid_lane_detection(
                    left_fitx, right_fitx, ploty, binary_warped.shape[1], binary_warped.shape[0],
                    left_detected, right_detected, leftx, rightx)

            if not detection_successful:
                return original_img, detection_successful

            # Crear una imagen para dibujar las líneas
            warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
            color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

            if left_detected:
                left_line_pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
                cv2.polylines(color_warp, np.int32([left_line_pts]), isClosed=False, color=(255, 0, 0), thickness=10)

            if right_detected:
                right_line_pts = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
                cv2.polylines(color_warp, np.int32([right_line_pts]), isClosed=False, color=(0, 0, 255), thickness=10)

            # Volver a proyectar la imagen a la perspectiva original usando la matriz inversa (Minv)
            newwarp = cv2.warpPerspective(color_warp, Minv, (original_img.shape[1], original_img.shape[0]))

            # Combinar el resultado con la imagen original
            result = cv2.addWeighted(original_img, 1, newwarp, 0.3, 0)

            return result, detection_successful
        except Exception as e:
            print(f"Error en fit_polynomial: {e}")
            return original_img, False

    # Función de supervisión para validar la detección de carriles
    def is_valid_lane_detection(self, left_fitx, right_fitx, ploty, img_width, img_height,
                                left_detected, right_detected, leftx, rightx):
        # Número máximo de píxeles umbral
        max_lane_pixels = 15000  # Ajustar según tus observaciones

        # Manejar los casos cuando solo una línea de carril es detectada
        if left_detected and right_detected:
            # Comprobar el conteo máximo de píxeles para carriles izquierdo y derecho
            if len(leftx) > max_lane_pixels or len(rightx) > max_lane_pixels:
                return False

            lane_widths = right_fitx - left_fitx  # Ancho entre carriles en todas las posiciones y

            # Comprobar si los carriles se cruzan
            if np.any(lane_widths <= 0):
                return False

            # Comprobar si el ancho del carril está dentro del rango aceptable
            avg_lane_width = np.mean(lane_widths)

            # Definir rango aceptable de ancho de carril (en píxeles)
            min_lane_width = 0.4 * img_width  # Por ejemplo, 40% del ancho de la imagen
            max_lane_width = 0.8 * img_width  # Por ejemplo, 80% del ancho de la imagen

            if not (min_lane_width < avg_lane_width < max_lane_width):
                return False

            # Comprobar si los carriles están centrados dentro de un margen
            lane_center = (left_fitx[-1] + right_fitx[-1]) / 2  # En la parte inferior de la imagen
            image_center = img_width / 2
            center_offset = abs(lane_center - image_center)

            # Definir desplazamiento de centro aceptable (en píxeles)
            max_center_offset = 0.1 * img_width  # Por ejemplo, 10% del ancho de la imagen

            if center_offset > max_center_offset:
                return False

            # Si todas las comprobaciones pasan
            return True

        elif left_detected or right_detected:
            # Comprobar el conteo máximo de píxeles para la línea de carril detectada
            if left_detected and len(leftx) > max_lane_pixels:
                return False
            if right_detected and len(rightx) > max_lane_pixels:
                return False

            # Si solo una línea de carril es detectada, podemos aceptarla
            return True
        else:
            # No se detectaron líneas de carril
            return False

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()

    # Para deshabilitar la función de supervisión para depuración, establece strict_validation en False
    # mainWindow.strict_validation = False  # Descomenta esta línea para deshabilitar la supervisión

    mainWindow.show()
    sys.exit(app.exec_())
