import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import cv2
import sys
import os
import argparse

ap = argparse.ArgumentParser()
(args, rest) = ap.parse_known_args()

ap.add_argument("-cy", "--cythonmode", action="store_true",
    help = "skompiluj specjalną wersję kodu dla dodatkowej wydajności")

args = ap.parse_args(rest)

import windowView
if args.cythonmode:
    import pyximport; pyximport.install(language_level=3, setup_args={'include_dirs': np.get_include()})
    import yolodetect
else:
    import yolodetectog as yolodetect


class Stream(QObject):
    newText = pyqtSignal(str)

    def write(self, text):
        self.newText.emit(str(text))


class MainDialog(QMainWindow, windowView.Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainDialog, self).__init__(parent)
        self.setupUi(self)
        self.vs = None
        self.confidence = 0.5
        self.threshold = 0.3
        self.outputfile = None
        self.fileonly = False
        self.drawpaths = False
        self.started = False
        self.maxloss = 1000
        self.logging = False
        self.calccross = False

        sys.stdout = Stream(newText=self.onUpdateText)

        self.pathEdit.textEdited.connect(self.pathEdited)
        self.oPathEdit.textEdited.connect(self.opathEdited)
        self.confidenceSpinBox.valueChanged.connect(self.confidenceChanged)
        self.thresholdSpinBox.valueChanged.connect(self.thresholdChanged)
        self.fileonlyCheckBox.clicked.connect(self.fileonlyClicked)
        self.drawpathsCheckBox.clicked.connect(self.drawpathsClicked)
        self.maxlossSpinBox.valueChanged.connect(self.maxlossChanged)
        self.calccrossCheckBox.clicked.connect(self.calccrossClicked)
        self.loggingCheckBox.clicked.connect(self.loggingClicked)
        self.startButton.clicked.connect(self.startClicked)
        self.browseButton.clicked.connect(self.browseClicked)

    def onUpdateText(self, text):
        cursor = self.textOutput.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.textOutput.setTextCursor(cursor)
        self.textOutput.ensureCursorVisible()

    def pathEdited(self):
        if self.sender().text().isspace() or self.sender().text() == "":
            self.startButton.setEnabled(False)
        else:
            self.startButton.setEnabled(True)

        self.vs = self.sender().text()
        #print("vs: {}".format(self.vs))

    def opathEdited(self):
        if self.sender().text().isspace() or self.sender().text() == "":
            self.fileonlyLabel.setEnabled(False)
            self.fileonlyCheckBox.setEnabled(False)
        else:
            self.fileonlyLabel.setEnabled(True)
            self.fileonlyCheckBox.setEnabled(True)
            self.outputfile = self.sender().text()
            #print("outputfile: {}".format(self.outputfile))

    def confidenceChanged(self):
        self.confidence = round(self.confidenceSpinBox.value(), 2)
        #print("confidence: {}".format(self.confidence))

    def thresholdChanged(self):
        self.threshold = round(self.thresholdSpinBox.value(), 2)
        #print("threshold: {}".format(self.threshold))

    def fileonlyClicked(self):
        if self.fileonly:
            self.fileonly = False
            #print("fileonly: False")
        else:
            self.fileonly = True
            #print("fileonly: True")

    def drawpathsClicked(self):
        if self.drawpaths:
            self.drawpaths = False
            self.maxlossLabel.setEnabled(False)
            self.maxlossSpinBox.setEnabled(False)
            self.calccrossLabel.setEnabled(False)
            self.calccrossCheckBox.setEnabled(False)
            self.calccrossCheckBox.setChecked(False)
            self.calccross = False
            #print("drawpaths: False")
        else:
            self.drawpaths = True
            self.maxlossLabel.setEnabled(True)
            self.maxlossSpinBox.setEnabled(True)
            self.calccrossLabel.setEnabled(True)
            self.calccrossCheckBox.setEnabled(True)
            #print("drawpaths: True")

    def maxlossChanged(self):
        self.maxloss = self.maxlossSpinBox.value()
    
    def calccrossClicked(self):
        if self.calccross:
            self.calccross = False
        else:
            self.calccross = True

    def loggingClicked(self):
        if self.logging:
            self.logging = False
        else:
            self.logging = True

    def browseClicked(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        cwd = os.getcwd()
        fname = None
        fname = QFileDialog.getOpenFileName(self, "Otwórz plik", cwd, "Pliki wideo (*.mpg *.mp4 *.ts)")
        if not fname:
            return
        self.pathEdit.setText(fname[0])
        self.pathEdited()
        self.vs = fname[0]

    def startClicked(self):
        if self.started:
            yolodetect.stopSignal = True
            print("[INFO] Zakończono działanie detekcji.")
            return
        if not self.vs:
            print("[BŁĄD] Nie podano nazwy pliku!")
            return
        if not os.path.isfile(self.vs):
            print("[BŁĄD] Podana nazwa pliku nie istnieje!")
            return
            
        yolodetect.stopSignal = False
        vs = cv2.VideoCapture(self.vs)
        totalframes = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
        self.progressBar.setMaximum(totalframes)
        self.progressBar.setValue(0)
        self.started = True
        self.startButton.setText("Stop")
        self.startButton.setStyleSheet("color: red")
        yolodetect.startDetect(self.vs ,vs, self.confidence, self.threshold, self.outputfile, self.fileonly, self.drawpaths,
                            self.maxloss, self.logging, self.calccross, True, self.progressBar)
        self.started = False
        self.startButton.setText("Start")
        self.startButton.setStyleSheet("color : black")

    def closeEvent(self, event):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        super().closeEvent(event)

