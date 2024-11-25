import sys
import cv2
import yaml
import numpy as np
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox, QGraphicsScene,
    QGraphicsEllipseItem, QGraphicsItem, QGraphicsView
)
from PyQt5.QtGui import QImage, QPixmap, QIcon, QBrush, QColor
from PyQt5.QtCore import QTimer, Qt, QPointF
from pathlib import Path
from gui8 import Ui_MainWindow

class Preprocessing:
    def __init__(self, s_thresh=(170, 255), sx_thresh=(20, 100)):
        self.s_thresh = s_thresh
        self.sx_thresh = sx_thresh

    def s_channel_threshold(self, img):
        # Convertir a espacio de color HLS
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        s_channel = hls[:, :, 2]

        # Aplicar umbral al canal S
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= self.s_thresh[0]) & (s_channel <= self.s_thresh[1])] = 1

        return s_binary

    def sobel_threshold(self, img):
        # Convertir a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Aplicar filtro Sobel en el eje x
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)

        # Escalar a rango de 0 a 255
        abs_sobelx = np.absolute(sobelx)
        if np.max(abs_sobelx) == 0:
            scaled_sobel = np.zeros_like(abs_sobelx, dtype=np.uint8)
        else:
            scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        # Aplicar umbral al gradiente x
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= self.sx_thresh[0]) & (scaled_sobel <= self.sx_thresh[1])] = 1

        return sxbinary

class Processing:
    def __init__(self, nwindows=9, margin=100):
        self.nwindows = nwindows
        self.margin = margin

        # Puntos de origen y destino para la transformación de perspectiva
        self.default_src = np.float32([
            [200, 720],
            [1100, 720],
            [595, 450],
            [685, 450]
        ])
        self.src = self.default_src.copy()
        self.dst = np.float32([
            [300, 720],
            [980, 720],
            [300, 0],
            [980, 0]
        ])

    def perspective_transform(self, img):
        img_size = (img.shape[1], img.shape[0])
        M = cv2.getPerspectiveTransform(self.src, self.dst)
        Minv = cv2.getPerspectiveTransform(self.dst, self.src)
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
        return warped, Minv

    def fit_polynomial(self, binary_warped, draw_windows=False):
        # Asegurar que binary_warped es de tipo uint8
        binary_warped = binary_warped.astype(np.uint8)

        # Tomar el histograma del tercio inferior de la imagen
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)

        # Crear una salida para dibujar
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

        # Encontrar los picos en el histograma que representan las líneas
        midpoint = int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Configuración de ventanas
        nwindows = self.nwindows
        window_height = int(binary_warped.shape[0]//nwindows)
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Posiciones actuales para actualizar
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Variables para guardar los índices de píxeles de izquierda y derecha
        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            # Definir límites de ventanas
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin

            # Dibujar ventanas
            if draw_windows:
                cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                              (win_xleft_high, win_y_high), (0, 255, 0), 2)
                cv2.rectangle(out_img, (win_xright_low, win_y_low),
                              (win_xright_high, win_y_high), (0, 255, 0), 2)

            # Identificar los píxeles no cero en x e y dentro de la ventana
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # Agregar índices a las listas
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # Si se encuentran más de minpix píxeles, recenter la siguiente ventana
            minpix = 50
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        # Concatenar los arrays de índices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extraer las posiciones de píxeles de izquierda y derecha
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Verificar si se encontraron píxeles suficientes para ajustar una línea
        if len(leftx) == 0 or len(lefty) == 0 or len(rightx) == 0 or len(righty) == 0:
            return None, None, None, None, None, out_img

        # Ajustar una curva polinomial de segundo grado
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generar puntos para plotear
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        try:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        except TypeError:
            left_fitx = ploty * 0
            right_fitx = ploty * 0

        # Dibujar los píxeles detectados de las líneas en la imagen de salida
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Dibujar las líneas polinomiales si draw_windows es True
        if draw_windows:
            for index in range(len(left_fitx)):
                cv2.circle(out_img, (int(left_fitx[index]), int(ploty[index])), 2, (255, 255, 0), -1)
                cv2.circle(out_img, (int(right_fitx[index]), int(ploty[index])), 2, (255, 255, 0), -1)

        return left_fitx, right_fitx, ploty, left_fit, right_fit, out_img

class AlertGeneration:
    def __init__(self, min_lane_width_ratio=0.4, max_lane_width_ratio=0.8,
                 max_center_offset_ratio=0.1, departure_threshold_pixels=50):
        self.min_lane_width_ratio = min_lane_width_ratio
        self.max_lane_width_ratio = max_lane_width_ratio
        self.max_center_offset_ratio = max_center_offset_ratio
        self.departure_threshold_pixels = departure_threshold_pixels

    def is_valid_lane_detection(self, left_fitx, right_fitx, ploty, img_width, img_height):
        # Calcular el ancho del carril en la base de la imagen
        if left_fitx is not None and right_fitx is not None:
            lane_width_pixels = right_fitx[-1] - left_fitx[-1]
            lane_width_ratio = lane_width_pixels / img_width

            # Validar el ancho del carril
            if not (self.min_lane_width_ratio <= lane_width_ratio <= self.max_lane_width_ratio):
                return False

            # Calcular el desplazamiento del centro
            lane_center = (left_fitx[-1] + right_fitx[-1]) / 2
            center_offset_pixels = abs((img_width / 2) - lane_center)
            center_offset_ratio = center_offset_pixels / (img_width / 2)

            # Validar el desplazamiento del centro
            if center_offset_ratio > self.max_center_offset_ratio:
                return False

            return True
        else:
            return False

    def detect_lane_departure(self, left_fitx, right_fitx, ploty, img_width):
        # Calcular el centro del carril
        lane_center = (left_fitx[-1] + right_fitx[-1]) / 2
        vehicle_center = img_width / 2
        center_offset_pixels = lane_center - vehicle_center

        # Determinar si hay desviación hacia la izquierda o derecha
        deviation = center_offset_pixels

        # Definir umbral para considerar desviación significativa
        threshold = self.departure_threshold_pixels

        # Inicializar colores de las líneas
        color_left = (255, 0, 0)  # Azul en BGR
        color_right = (255, 0, 0)  # Azul en BGR

        # Si hay desviación, cambiar el color del lado correspondiente a rojo
        if deviation > threshold:
            # Desviación hacia la izquierda (vehículo se acerca al carril derecho)
            color_right = (0, 0, 255)  # Rojo
        elif deviation < -threshold:
            # Desviación hacia la derecha (vehículo se acerca al carril izquierdo)
            color_left = (0, 0, 255)  # Rojo

        return color_left, color_right

class DraggablePoint(QGraphicsEllipseItem):
    def __init__(self, x, y, index, parent=None):
        super().__init__(-5, -5, 10, 10, parent)
        self.setBrush(QBrush(Qt.red))
        self.setFlags(
            QGraphicsItem.ItemIsMovable |
            QGraphicsItem.ItemIsSelectable |
            QGraphicsItem.ItemSendsScenePositionChanges
        )
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.OpenHandCursor)
        self.setPos(x, y)
        self.index = index  # Índice del punto (0 a 3)
        self.setZValue(1)  # Asegura que los puntos estén sobre la imagen

    def mousePressEvent(self, event):
        self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self.setCursor(Qt.OpenHandCursor)
        super().mouseReleaseEvent(event)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange and self.scene():
            # Restringir el movimiento dentro de la escena
            new_pos = value
            rect = self.scene().sceneRect()
            new_x = min(max(new_pos.x(), rect.left()), rect.right())
            new_y = min(max(new_pos.y(), rect.top()), rect.bottom())
            return QPointF(new_x, new_y)
        return super().itemChange(change, value)

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # Inicializar atributos
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.scene = QGraphicsScene()
        self.graphicsView.setScene(self.scene)
        self.config = {}
        self.preprocessing = None
        self.processing = None
        self.alert_generation = None

        self.strict_validation = True  # Monitoreo de validación estricto

        # Atributos para manejar los modos de depuración
        self.debug_mode = 'fullButton'  # Valor por defecto
        self.debug_modes = {
            'fullButton': 'fullCode',
            'HSLButton': 'HSL',
            'SobelButton': 'Sobel',
            'fullPreButton': 'fullPre',
            'warpButton': 'warp',
            'slidingWindowButton': 'slidingWindow',
            'resultantButton': 'resultant',
        }

        # Atributo para manejar el pixmap
        self.pixmap_item = None

        # Cargar configuración
        self.load_config()

        # Conectar señales
        self.startButton.clicked.connect(self.start_video)
        self.stopButton.clicked.connect(self.stop_video)
        self.picBrowse.clicked.connect(self.picFileDialog)
        self.vidBrowse.clicked.connect(self.vidFileDialog)
        self.applyButton.clicked.connect(self.applyConfig)
        self.cancelButton.clicked.connect(self.go_to_main_tab)
        self.visualBox.stateChanged.connect(self.toggleVisionIcon)
        self.soundBox.stateChanged.connect(self.toggleSoundIcon)

        # Conectar radioButtons de depuración
        self.fullButton.toggled.connect(self.update_debug_mode)
        self.HSLButton.toggled.connect(self.update_debug_mode)
        self.SobelButton.toggled.connect(self.update_debug_mode)
        self.fullPreButton.toggled.connect(self.update_debug_mode)
        self.warpButton.toggled.connect(self.update_debug_mode)
        self.slidingWindowButton.toggled.connect(self.update_debug_mode)
        self.resultantButton.toggled.connect(self.update_debug_mode)

        # Nuevos checkboxes
        self.supervisionBox.stateChanged.connect(self.update_supervision_mode)
        self.shSupervisionBox.stateChanged.connect(self.update_sh_supervision_mode)
        self.visualThreshBox.stateChanged.connect(self.update_visual_thresh_mode)
        self.editCheck.stateChanged.connect(self.toggle_edit_mode)  # Conectar el checkbox de edición

        # Configurar el QGraphicsView
        self.graphicsView.setDragMode(QGraphicsView.NoDrag)
        self.graphicsView.setFocusPolicy(Qt.StrongFocus)

        # Estados iniciales
        self.supervision_enabled = True
        self.sh_supervision_enabled = False
        self.visual_thresh_enabled = False
        self.edit_mode = False

        # Puntos de edición
        self.edit_points = []
        self.current_frame = None  # Para almacenar el frame actual

        # Iniciar en mainTab por defecto
        self.tabs.setCurrentIndex(0)

    def load_config(self):
        config_file = 'settings.yaml'
        try:
            with open(config_file, 'r') as yaml_file:
                self.config = yaml.safe_load(yaml_file)
        except FileNotFoundError:
            # Si no existe el archivo, inicializamos la configuración por defecto
            self.config = {}
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al cargar el archivo de configuración: {e}")
            self.config = {}

        # Comprobar y actualizar la configuración
        config_changed = self.check_and_update_config()
        if config_changed:
            self.save_config()

        # Actualizar la GUI con los valores de configuración
        self.update_gui_from_config()

    def check_and_update_config(self):
        config_changed = False

        # Verificar y actualizar las secciones necesarias
        default_config = {
            'dataAcq': {
                'dataSource': 'Camera',
                'picSource': '',
                'vidSource': ''
            },
            'alertConf': {
                'visualAlert': 'OFF',
                'soundAlert': 'OFF'
            },
            'preproConf': {
                'SChannelThreshold': {
                    'minimum': 170,
                    'maximum': 255
                },
                'SobelThreshold': {
                    'minimum': 20,
                    'maximum': 100
                }
            },
            'proConf': {
                'nWindows': 9,
                'margin': 100,
                'laneWidth': {
                    'minimum': 0.4,
                    'maximum': 0.8
                },
                'maxCenterOffset': 0.1,
                'departureThreshold': 50
            },
            'debugConf': {
                'fullPreButton': False,
                'fullButton': True,
                'HSLButton': False,
                'SobelButton': False,
                'warpButton': False,
                'slidingWindowButton': False,
                'resultantButton': False,
            }
        }

        # Actualizar la configuración existente con los valores por defecto si faltan
        for key, value in default_config.items():
            if key not in self.config:
                self.config[key] = value
                config_changed = True
            else:
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if subkey not in self.config[key]:
                            self.config[key][subkey] = subvalue
                            config_changed = True
                        else:
                            if isinstance(subvalue, dict):
                                for subsubkey, subsubvalue in subvalue.items():
                                    if subsubkey not in self.config[key][subkey]:
                                        self.config[key][subkey][subsubkey] = subsubvalue
                                        config_changed = True

        return config_changed

    def update_gui_from_config(self):
        # Actualizar los campos de dataAcq
        self.picEdit.setText(self.config['dataAcq']['picSource'])
        self.vidEdit.setText(self.config['dataAcq']['vidSource'])

        data_source = self.config['dataAcq']['dataSource']
        if data_source == 'Camera':
            self.camButton.setChecked(True)
        elif data_source == 'Picture':
            self.picButton.setChecked(True)
        elif data_source == 'Video':
            self.vidButton.setChecked(True)

        # Actualizar las casillas de verificación de alertas
        self.visualBox.setChecked(self.config['alertConf']['visualAlert'] == 'ON')
        self.soundBox.setChecked(self.config['alertConf']['soundAlert'] == 'ON')

        # Actualizar los campos de preproConf
        self.minSChEdit.setText(str(self.config['preproConf']['SChannelThreshold']['minimum']))
        self.maxSChEdit.setText(str(self.config['preproConf']['SChannelThreshold']['maximum']))
        self.minSobelEdit.setText(str(self.config['preproConf']['SobelThreshold']['minimum']))
        self.maxSobelEdit.setText(str(self.config['preproConf']['SobelThreshold']['maximum']))

        # Actualizar los campos de proConf
        self.nWindowEdit.setText(str(self.config['proConf']['nWindows']))
        self.marginEdit.setText(str(self.config['proConf']['margin']))
        self.minLaneWidthEdit.setText(str(self.config['proConf']['laneWidth']['minimum']))
        self.maxLaneWidthEdit.setText(str(self.config['proConf']['laneWidth']['maximum']))
        self.maxCenterEdit.setText(str(self.config['proConf']['maxCenterOffset']))
        self.departureEdit.setText(str(self.config['proConf']['departureThreshold']))

        # Actualizar los radioButtons de debugConf
        self.fullPreButton.setChecked(self.config['debugConf']['fullPreButton'])
        self.fullButton.setChecked(self.config['debugConf']['fullButton'])
        self.HSLButton.setChecked(self.config['debugConf']['HSLButton'])
        self.SobelButton.setChecked(self.config['debugConf']['SobelButton'])
        self.warpButton.setChecked(self.config['debugConf']['warpButton'])
        self.slidingWindowButton.setChecked(self.config['debugConf']['slidingWindowButton'])
        self.resultantButton.setChecked(self.config['debugConf']['resultantButton'])

        # Llamar a toggleVisionIcon y toggleSoundIcon para actualizar los iconos
        self.toggleVisionIcon(self.visualBox.checkState())
        self.toggleSoundIcon(self.soundBox.checkState())

    def save_config(self):
        config_file = 'settings.yaml'
        try:
            with open(config_file, 'w') as yaml_file:
                yaml.dump(self.config, yaml_file, default_flow_style=False)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al guardar el archivo de configuración: {e}")

    def start_video(self):
        data_source = self.config['dataAcq']['dataSource']
        if data_source == "Video":
            video_path = self.config['dataAcq']['vidSource']
            if video_path:
                if self.cap is not None:
                    self.cap.release()
                self.cap = cv2.VideoCapture(video_path)
                self.timer.start(30)
            else:
                QMessageBox.warning(self, "Advertencia", "No se ha especificado ningún archivo de video.")
        elif data_source == "Camera":
            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(0)
            self.timer.start(30)
        else:
            QMessageBox.warning(self, "Advertencia", "Fuente de datos no soportada para iniciar.")

        # Actualiza las clases de procesamiento con los valores actuales
        self.update_processing_classes()

    def stop_video(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.timer.stop()
        # Remover puntos de edición si existen
        if self.edit_points:
            for point in self.edit_points:
                if point.scene():
                    self.scene.removeItem(point)
            self.edit_points = []
        # Limpiar la escena y el pixmap_item
        self.scene.clear()
        self.pixmap_item = None
        self.edit_mode = False
        self.editCheck.setChecked(False)

    def go_to_main_tab(self):
        self.tabs.setCurrentIndex(0)

    def picFileDialog(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Seleccionar Imagen", "", "Archivos de imagen (*.jpg *.png *.jpeg)")
        if filename:
            path = Path(filename)
            self.picEdit.setText(str(path))
            self.config['dataAcq']['picSource'] = str(path)
            self.save_config()

            if self.cap is not None:
                self.cap.release()
                self.cap = None
                self.timer.stop()

            try:
                image_path = os.path.normpath(str(path))
                image_data = np.fromfile(image_path, dtype=np.uint8)
                img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                if img is not None:
                    # Actualiza las clases de procesamiento con los valores actuales
                    self.update_processing_classes()
                    self.current_frame = img.copy()
                    processed_img = self.lane_detection_pipeline(img)
                    self.display_image(processed_img)
                else:
                    QMessageBox.critical(self, "Error", "Error al cargar la imagen.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error al cargar la imagen: {e}")

    def vidFileDialog(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Seleccionar Video", "", "Archivos de video (*.mp4 *.mpeg *.avi)")
        if filename:
            path = Path(filename)
            self.vidEdit.setText(str(path))
            self.config['dataAcq']['vidSource'] = str(path)
            self.save_config()

            if self.cap is not None:
                self.cap.release()

            self.cap = cv2.VideoCapture(str(path))
            self.timer.start(30)

            # Actualiza las clases de procesamiento con los valores actuales
            self.update_processing_classes()

    def toggleVisionIcon(self, state):
        if state == Qt.Checked:
            self.visualBox.setIcon(QIcon(os.path.join('imgss', "visionON.svg")))
            self.vision_enabled = True
            self.config['alertConf']['visualAlert'] = 'ON'
        else:
            self.visualBox.setIcon(QIcon(os.path.join('imgss', "visionOFF.svg")))
            self.vision_enabled = False
            self.config['alertConf']['visualAlert'] = 'OFF'
        self.save_config()

    def toggleSoundIcon(self, state):
        if state == Qt.Checked:
            self.soundBox.setIcon(QIcon(os.path.join('imgss', "soundON.svg")))
            self.sound_enabled = True
            self.config['alertConf']['soundAlert'] = 'ON'
        else:
            self.soundBox.setIcon(QIcon(os.path.join('imgss', "soundOFF.svg")))
            self.sound_enabled = False
            self.config['alertConf']['soundAlert'] = 'OFF'
        self.save_config()

    def applyConfig(self):
        # Guardar configuración directamente en 'settings.yaml' sin ventana emergente
        # Actualizamos self.config con los valores actuales de los campos de la GUI
        self.config['dataAcq']['dataSource'] = self.getDataSource()
        self.config['dataAcq']['picSource'] = self.picEdit.text()
        self.config['dataAcq']['vidSource'] = self.vidEdit.text()

        self.config['alertConf']['visualAlert'] = 'ON' if self.visualBox.isChecked() else 'OFF'
        self.config['alertConf']['soundAlert'] = 'ON' if self.soundBox.isChecked() else 'OFF'

        self.config['preproConf']['SChannelThreshold']['minimum'] = int(self.minSChEdit.text())
        self.config['preproConf']['SChannelThreshold']['maximum'] = int(self.maxSChEdit.text())

        self.config['preproConf']['SobelThreshold']['minimum'] = int(self.minSobelEdit.text())
        self.config['preproConf']['SobelThreshold']['maximum'] = int(self.maxSobelEdit.text())

        self.config['proConf']['nWindows'] = int(self.nWindowEdit.text())
        self.config['proConf']['margin'] = int(self.marginEdit.text())
        self.config['proConf']['laneWidth']['minimum'] = float(self.minLaneWidthEdit.text())
        self.config['proConf']['laneWidth']['maximum'] = float(self.maxLaneWidthEdit.text())
        self.config['proConf']['maxCenterOffset'] = float(self.maxCenterEdit.text())
        self.config['proConf']['departureThreshold'] = int(self.departureEdit.text())

        # Actualizar debugConf
        self.config['debugConf'] = {
            'fullPreButton': self.fullPreButton.isChecked(),
            'fullButton': self.fullButton.isChecked(),
            'HSLButton': self.HSLButton.isChecked(),
            'SobelButton': self.SobelButton.isChecked(),
            'warpButton': self.warpButton.isChecked(),
            'slidingWindowButton': self.slidingWindowButton.isChecked(),
            'resultantButton': self.resultantButton.isChecked(),
        }

        # Guardamos la configuración
        self.save_config()

        QMessageBox.information(self, "Configuración", "La configuración ha sido guardada en settings.yaml.")

        # Actualizar el modo de depuración
        self.update_debug_mode()

    def getDataSource(self):
        if self.camButton.isChecked():
            return "Camera"
        elif self.picButton.isChecked():
            return "Picture"
        elif self.vidButton.isChecked():
            return "Video"
        else:
            return "Unknown"

    def update_frame(self):
        if self.cap is not None and self.cap.isOpened():
            # Si el modo de edición está activo, no actualizar el frame
            if self.edit_mode:
                return

            ret, frame = self.cap.read()
            if ret:
                # Actualiza las clases de procesamiento con los valores actuales
                self.update_processing_classes()
                self.current_frame = frame.copy()
                processed_frame = self.lane_detection_pipeline(frame)
                self.display_image(processed_frame)
            else:
                self.cap.release()
                self.cap = None
                self.timer.stop()
        else:
            self.timer.stop()

    def display_image(self, img):
        try:
            if len(img.shape) == 2:
                # Imagen en escala de grises
                height, width = img.shape
                bytes_per_line = width
                q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            else:
                # Asegurar que img es de tipo uint8
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                height, width, channel = img.shape
                bytes_per_line = 3 * width
                img = np.require(img, np.uint8, 'C')
                q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)

            pixmap = QPixmap.fromImage(q_img)

            if self.pixmap_item is None:
                self.pixmap_item = self.scene.addPixmap(pixmap)
                self.pixmap_item.setZValue(0)  # Asegurar que la imagen está detrás de los puntos
            else:
                self.pixmap_item.setPixmap(pixmap)

            # Establecer el tamaño de la escena al tamaño de la imagen
            self.scene.setSceneRect(0, 0, width, height)

            # Si está en modo edición, agregar los puntos
            if self.edit_mode:
                self.add_edit_points()

            self.graphicsView.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error en display_image: {e}")

    def add_edit_points(self):
        # Solo agregar los puntos si no están ya agregados
        if not self.edit_points:
            # Agregar puntos interactivos
            for i, (x, y) in enumerate(self.processing.src):
                point = DraggablePoint(x, y, i)
                self.edit_points.append(point)
                self.scene.addItem(point)
        else:
            # Actualizar posiciones de los puntos existentes
            for i, point in enumerate(self.edit_points):
                point.setPos(*self.processing.src[i])

    def update_processing_classes(self):
        # Valores de preprocesamiento
        s_thresh = (
            self.config['preproConf']['SChannelThreshold']['minimum'],
            self.config['preproConf']['SChannelThreshold']['maximum']
        )
        sx_thresh = (
            self.config['preproConf']['SobelThreshold']['minimum'],
            self.config['preproConf']['SobelThreshold']['maximum']
        )
        self.preprocessing = Preprocessing(s_thresh, sx_thresh)

        # Valores de procesamiento
        nwindows = self.config['proConf']['nWindows']
        margin = self.config['proConf']['margin']
        if not hasattr(self, 'processing') or self.processing is None:
            self.processing = Processing(nwindows, margin)
        else:
            # Mantener los puntos src existentes
            self.processing.nwindows = nwindows
            self.processing.margin = margin

        # Valores de alerta
        min_lane_width_ratio = self.config['proConf']['laneWidth']['minimum']
        max_lane_width_ratio = self.config['proConf']['laneWidth']['maximum']
        max_center_offset_ratio = self.config['proConf']['maxCenterOffset']
        departure_threshold_pixels = self.config['proConf']['departureThreshold']
        self.alert_generation = AlertGeneration(min_lane_width_ratio, max_lane_width_ratio,
                                                max_center_offset_ratio, departure_threshold_pixels)

    def update_debug_mode(self):
        for key in self.debug_modes.keys():
            radio_button = getattr(self, key)
            if radio_button.isChecked():
                self.debug_mode = key
                break
        else:
            self.debug_mode = 'fullButton'  # Valor por defecto

    def update_supervision_mode(self, state):
        self.supervision_enabled = (state == Qt.Checked)

    def update_sh_supervision_mode(self, state):
        self.sh_supervision_enabled = (state == Qt.Checked)

    def update_visual_thresh_mode(self, state):
        self.visual_thresh_enabled = (state == Qt.Checked)

    def toggle_edit_mode(self, state):
        self.edit_mode = (state == Qt.Checked)
        if self.edit_mode:
            # Detener la actualización de frames mientras se edita
            self.timer.stop()
            # Mostrar la imagen con los puntos
            if self.current_frame is not None:
                self.display_image(self.current_frame)
        else:
            # Obtener las posiciones actuales de los puntos y actualizar src
            if self.edit_points:
                new_src = []
                for point in self.edit_points:
                    pos = point.pos()
                    new_src.append([pos.x(), pos.y()])
                self.processing.src = np.float32(new_src)
                # Remover puntos de la escena
                for point in self.edit_points:
                    if point.scene():
                        self.scene.removeItem(point)
                self.edit_points = []
            # Reanudar la actualización de frames
            if self.cap is not None:
                self.timer.start(30)

    def lane_detection_pipeline(self, frame):
        try:
            if self.preprocessing is None or self.processing is None or self.alert_generation is None:
                self.update_processing_classes()

            # Obtener el modo de depuración actual
            debug_mode = self.debug_modes.get(self.debug_mode, 'fullCode')

            # Preprocesamiento
            s_binary = self.preprocessing.s_channel_threshold(frame)
            sxbinary = self.preprocessing.sobel_threshold(frame)

            if debug_mode == 'HSL':
                # Mostrar la imagen del canal S
                return cv2.cvtColor(s_binary * 255, cv2.COLOR_GRAY2BGR)
            elif debug_mode == 'Sobel':
                # Mostrar la imagen después del filtro Sobel
                return cv2.cvtColor(sxbinary * 255, cv2.COLOR_GRAY2BGR)
            else:
                # Continuar con el pipeline
                combined_binary = np.zeros_like(s_binary)
                combined_binary[(s_binary == 1) | (sxbinary == 1)] = 255

            if debug_mode == 'fullPre':
                # Mostrar la combinación de HSL y Sobel
                return cv2.cvtColor(combined_binary, cv2.COLOR_GRAY2BGR)

            # Transformación de perspectiva
            warped, Minv = self.processing.perspective_transform(combined_binary)

            if debug_mode == 'warp':
                return cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)

            # Detección y ajuste de carriles
            draw_windows = (debug_mode == 'slidingWindow')

            left_fitx, right_fitx, ploty, left_fit, right_fit, out_img = self.processing.fit_polynomial(
                warped, draw_windows=draw_windows)

            if debug_mode == 'slidingWindow':
                # Aplicar la transformación inversa a out_img para mostrar en perspectiva original
                newwarp = cv2.warpPerspective(out_img, Minv, (frame.shape[1], frame.shape[0]))
                return newwarp

            if left_fitx is None or right_fitx is None:
                # No se detectaron carriles
                if debug_mode == 'resultant':
                    # En 'resultant', mostrar la imagen warpada sin líneas
                    return cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
                else:
                    return frame

            if self.strict_validation and self.supervision_enabled:
                detection_successful = self.alert_generation.is_valid_lane_detection(
                    left_fitx, right_fitx, ploty, frame.shape[1], frame.shape[0])
            else:
                detection_successful = True

            if not detection_successful:
                # No se detectó un carril válido
                if debug_mode == 'resultant':
                    # En 'resultant', mostrar la imagen warpada sin líneas
                    return cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
                else:
                    return frame

            # Crear una imagen en blanco para dibujar las líneas
            warp_zero = np.zeros_like(warped).astype(np.uint8)
            color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

            # Preparar los puntos para las líneas
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))

            if debug_mode == 'resultant':
                # Dibujar las líneas en color azul sin considerar la alerta de salida
                cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255, 0, 0), thickness=20)
                cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(255, 0, 0), thickness=20)

                # Agregar visualización de supervisión si está habilitada
                if self.sh_supervision_enabled:
                    lane_width_min = self.alert_generation.min_lane_width_ratio * frame.shape[1]
                    lane_width_max = self.alert_generation.max_lane_width_ratio * frame.shape[1]
                    # Dibujar líneas verticales para visualizar los umbrales
                    left_boundary = (frame.shape[1] - lane_width_max) / 2
                    right_boundary = frame.shape[1] - left_boundary
                    cv2.line(color_warp, (int(left_boundary), 0), (int(left_boundary), frame.shape[0]), (255, 0, 255), 5)
                    cv2.line(color_warp, (int(right_boundary), 0), (int(right_boundary), frame.shape[0]), (255, 0, 255), 5)

                # Agregar visualización de umbrales de salida si está habilitada
                if self.visual_thresh_enabled:
                    departure_threshold = self.alert_generation.departure_threshold_pixels
                    # Dibujar líneas verticales para visualizar los umbrales de salida
                    center = frame.shape[1] / 2
                    cv2.line(color_warp, (int(center - departure_threshold), 0),
                             (int(center - departure_threshold), frame.shape[0]), (0, 255, 255), 5)
                    cv2.line(color_warp, (int(center + departure_threshold), 0),
                             (int(center + departure_threshold), frame.shape[0]), (0, 255, 255), 5)

                return color_warp
            else:
                # Obtener los colores para las líneas considerando la desviación
                color_left, color_right = self.alert_generation.detect_lane_departure(
                    left_fitx, right_fitx, ploty, frame.shape[1])

                # Dibujar las líneas en color_warp
                cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=color_left, thickness=20)
                cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=color_right, thickness=20)

                # Aplicar la transformación inversa a color_warp
                newwarp = cv2.warpPerspective(color_warp, Minv, (frame.shape[1], frame.shape[0]))

                # Superponer las líneas en la imagen original
                result = cv2.addWeighted(frame, 1, newwarp, 1, 0)

                return result

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error en lane_detection_pipeline: {e}")
            return frame

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
