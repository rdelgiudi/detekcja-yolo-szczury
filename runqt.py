import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import sip

import signalcontroller


app = QApplication(sys.argv)
form = signalcontroller.MainDialog()
form.show()
app.exec_()