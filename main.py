from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QIcon
from pathlib import Path
from gui import Ui_MainWindow
import yaml

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

    def picFileDialog(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select a File", "", "Image files (*.jpg);;Image files (*.png)")
        if filename:
            path = Path(filename)
            self.picEdit.setText(str(path))

    def vidFileDialog(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select a File", "", "Video files (*.mp4);;Video files (*.mpeg)")
        if filename:
            path = Path(filename)
            self.vidEdit.setText(str(path))
            
    def toggleVisionIcon(self, state):
        if state == 2:
            self.visualBox.setIcon(QIcon("Imágenes/visionON.svg"))
        else:
            self.visualBox.setIcon(QIcon("Imágenes/visionOFF.svg"))

    def toggleSoundIcon(self, state):
        if state == 2:
            self.soundBox.setIcon(QIcon("Imágenes/soundON.svg"))
        else:
            self.soundBox.setIcon(QIcon("Imágenes/soundOFF.svg"))
    
    def applyConfig(self):
        file_name, _ = QFileDialog.getSaveFileName(self, 'Guardar archivo YAML', '', 'Archivos YAML (*.yaml)')
        if file_name:
            data = {
                'dataAcq':{
                    'dataSource': self.camButton.text() if self.camButton.isChecked() else
                                     self.picButton.text() if self.picButton.isChecked() else
                                     self.vidButton.text(),
                'picSource': self.picEdit.text(),
                'vidSource': self.vidEdit.text(),
                },
                'alertConf':{
                    'visualAlert': 'ON' if self.visualBox.isChecked() else 'OFF',
                    'soundAlert': 'ON' if self.soundBox.isChecked() else 'OFF',
                },
                'senseConf':{
                    'visualSense': self.visualSenList.currentText(),
                    'soundSense': self.soundSenList.currentText(),
                },
                'camCalib':{
                    'focalLength': self.focalEdit.text(),
                    'imageCenterPoint': self.icpEdit.text(),
                    'efectivePixelSize': self.epsLine.text(),
                    'radialDistortion':{
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
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())