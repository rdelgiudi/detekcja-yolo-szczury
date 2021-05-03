from PyQt5.QtWidgets import QProgressBar

def setupBar(bar: QProgressBar, totalframes):
    bar.setValue(1)
    bar.setMaximum(totalframes)
    return bar

def updateBar(bar: QProgressBar, val):
    bar.setValue(val)
