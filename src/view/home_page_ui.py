# Form implementation generated from reading ui file 'c:\Users\hokag\Documents\GitHub\BKAI-Demo-Motorbike-PyQT\src\view\home_page.ui'
#
# Created by: PyQt6 UI code generator 6.4.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_HomePage(object):
    def setupUi(self, HomePage):
        HomePage.setObjectName("HomePage")
        HomePage.resize(1342, 757)
        self.centralwidget = QtWidgets.QWidget(parent=HomePage)
        self.centralwidget.setObjectName("centralwidget")
        self.frame = QtWidgets.QFrame(parent=self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(0, -10, 1321, 741))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.frame.setFont(font)
        self.frame.setStyleSheet("")
        self.frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.frame.setObjectName("frame")
        self.spin_label = QtWidgets.QLabel(parent=self.frame)
        self.spin_label.setGeometry(QtCore.QRect(1290, 60, 31, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setItalic(True)
        self.spin_label.setFont(font)
        self.spin_label.setText("")
        self.spin_label.setPixmap(QtGui.QPixmap(":/icons/loading.gif"))
        self.spin_label.setObjectName("spin_label")
        self.progressBar = QtWidgets.QProgressBar(parent=self.frame)
        self.progressBar.setGeometry(QtCore.QRect(690, 64, 571, 23))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.progressBar.setFont(font)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.progressBar.setTextVisible(False)
        self.progressBar.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.progressBar.setObjectName("progressBar")
        self.label_6 = QtWidgets.QLabel(parent=self.frame)
        self.label_6.setGeometry(QtCore.QRect(570, 53, 111, 41))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setItalic(True)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(parent=self.frame)
        self.label_7.setGeometry(QtCore.QRect(60, 90, 131, 41))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setItalic(True)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.input_button = QtWidgets.QToolButton(parent=self.frame)
        self.input_button.setGeometry(QtCore.QRect(460, 131, 21, 28))
        self.input_button.setToolTipDuration(-3)
        self.input_button.setStyleSheet("")
        self.input_button.setObjectName("input_button")
        self.input_lineEdit = QtWidgets.QLineEdit(parent=self.frame)
        self.input_lineEdit.setGeometry(QtCore.QRect(60, 132, 401, 26))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.input_lineEdit.setFont(font)
        self.input_lineEdit.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.IBeamCursor))
        self.input_lineEdit.setStyleSheet("")
        self.input_lineEdit.setText("")
        self.input_lineEdit.setReadOnly(True)
        self.input_lineEdit.setObjectName("input_lineEdit")
        self.label_8 = QtWidgets.QLabel(parent=self.frame)
        self.label_8.setGeometry(QtCore.QRect(60, 190, 161, 41))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setItalic(True)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.output_button = QtWidgets.QToolButton(parent=self.frame)
        self.output_button.setGeometry(QtCore.QRect(460, 230, 21, 28))
        self.output_button.setToolTipDuration(-3)
        self.output_button.setStyleSheet("")
        self.output_button.setObjectName("output_button")
        self.output_lineEdit = QtWidgets.QLineEdit(parent=self.frame)
        self.output_lineEdit.setGeometry(QtCore.QRect(60, 231, 401, 26))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.output_lineEdit.setFont(font)
        self.output_lineEdit.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.IBeamCursor))
        self.output_lineEdit.setStyleSheet("")
        self.output_lineEdit.setText("")
        self.output_lineEdit.setReadOnly(True)
        self.output_lineEdit.setPlaceholderText("")
        self.output_lineEdit.setObjectName("output_lineEdit")
        self.process_button = QtWidgets.QPushButton(parent=self.frame)
        self.process_button.setGeometry(QtCore.QRect(220, 560, 101, 41))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        self.process_button.setFont(font)
        self.process_button.setStyleSheet("QPushButton{border-radius:5px;background-color:rgb(0,\n"
"                            104,\n"
"                            74);color:white;font-weight:600}QPushButton:hover{border-radius:5px;border:2px\n"
"                            solid\n"
"                            rgb(123, 255, 244);background-color:rgb(0, 104,\n"
"                            74);color:white;font-weight:600}")
        self.process_button.setObjectName("process_button")
        self.label = QtWidgets.QLabel(parent=self.frame)
        self.label.setGeometry(QtCore.QRect(60, 40, 321, 41))
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.log_area = QtWidgets.QTextEdit(parent=self.frame)
        self.log_area.setGeometry(QtCore.QRect(550, 100, 771, 151))
        self.log_area.setReadOnly(True)
        self.log_area.setObjectName("log_area")
        self.line = QtWidgets.QFrame(parent=self.frame)
        self.line.setGeometry(QtCore.QRect(60, 280, 421, 20))
        self.line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line.setObjectName("line")
        self.line_2 = QtWidgets.QFrame(parent=self.frame)
        self.line_2.setGeometry(QtCore.QRect(520, 50, 20, 691))
        self.line_2.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line_2.setObjectName("line_2")
        self.device_label = QtWidgets.QLabel(parent=self.frame)
        self.device_label.setGeometry(QtCore.QRect(60, 290, 421, 41))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setItalic(False)
        self.device_label.setFont(font)
        self.device_label.setObjectName("device_label")
        self.conf_slide = QtWidgets.QSlider(parent=self.frame)
        self.conf_slide.setGeometry(QtCore.QRect(210, 350, 211, 16))
        self.conf_slide.setMinimum(1)
        self.conf_slide.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.conf_slide.setObjectName("conf_slide")
        self._label = QtWidgets.QLabel(parent=self.frame)
        self._label.setGeometry(QtCore.QRect(60, 336, 141, 41))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setItalic(False)
        self._label.setFont(font)
        self._label.setObjectName("_label")
        self.conf_label = QtWidgets.QLabel(parent=self.frame)
        self.conf_label.setGeometry(QtCore.QRect(440, 390, 41, 41))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setItalic(False)
        self.conf_label.setFont(font)
        self.conf_label.setText("")
        self.conf_label.setObjectName("conf_label")
        self.opt_light_enhance = QtWidgets.QCheckBox(parent=self.frame)
        self.opt_light_enhance.setGeometry(QtCore.QRect(60, 430, 161, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.opt_light_enhance.setFont(font)
        self.opt_light_enhance.setObjectName("opt_light_enhance")
        self.label_2 = QtWidgets.QLabel(parent=self.frame)
        self.label_2.setGeometry(QtCore.QRect(60, 390, 91, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.line_3 = QtWidgets.QFrame(parent=self.frame)
        self.line_3.setGeometry(QtCore.QRect(60, 370, 421, 20))
        self.line_3.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line_3.setObjectName("line_3")
        self.opt_fog_dehaze = QtWidgets.QCheckBox(parent=self.frame)
        self.opt_fog_dehaze.setGeometry(QtCore.QRect(340, 430, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.opt_fog_dehaze.setFont(font)
        self.opt_fog_dehaze.setObjectName("opt_fog_dehaze")
        self.widget = QtWidgets.QWidget(parent=self.frame)
        self.widget.setGeometry(QtCore.QRect(550, 270, 771, 431))
        self.widget.setObjectName("widget")
        self.pushButton = QtWidgets.QPushButton(parent=self.frame)
        self.pushButton.setGeometry(QtCore.QRect(910, 710, 75, 24))
        self.pushButton.setObjectName("pushButton")
        HomePage.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(parent=HomePage)
        self.statusbar.setObjectName("statusbar")
        HomePage.setStatusBar(self.statusbar)

        self.retranslateUi(HomePage)
        QtCore.QMetaObject.connectSlotsByName(HomePage)

    def retranslateUi(self, HomePage):
        _translate = QtCore.QCoreApplication.translate
        HomePage.setWindowTitle(_translate("HomePage", "Motorcycle Demo"))
        self.label_6.setText(_translate("HomePage", "Running Status"))
        self.label_7.setText(_translate("HomePage", "Choose input file:"))
        self.input_button.setToolTip(_translate("HomePage", "\n"
"                            <html><head/><body><p><br/></p></body></html>"))
        self.input_button.setWhatsThis(_translate("HomePage", "\n"
"                            <html><head/><body><p><br/></p></body></html>"))
        self.input_button.setText(_translate("HomePage", "..."))
        self.input_lineEdit.setToolTip(_translate("HomePage", "\n"
"                            <html><head/><body><p><br/></p></body></html>"))
        self.input_lineEdit.setPlaceholderText(_translate("HomePage", "(.mp4, .avi, .flv, .h265)"))
        self.label_8.setText(_translate("HomePage", "Choose output folder:"))
        self.output_button.setToolTip(_translate("HomePage", "\n"
"                            <html><head/><body><p><br/></p></body></html>"))
        self.output_button.setWhatsThis(_translate("HomePage", "\n"
"                            <html><head/><body><p><br/></p></body></html>"))
        self.output_button.setText(_translate("HomePage", "..."))
        self.output_lineEdit.setToolTip(_translate("HomePage", "\n"
"                            <html><head/><body><p><br/></p></body></html>"))
        self.process_button.setText(_translate("HomePage", "Process"))
        self.label.setText(_translate("HomePage", "Motorcycle Detection"))
        self.device_label.setText(_translate("HomePage", "Device:"))
        self._label.setText(_translate("HomePage", "Detect Confidence:"))
        self.opt_light_enhance.setText(_translate("HomePage", "Light Enhancement"))
        self.label_2.setText(_translate("HomePage", "Options"))
        self.opt_fog_dehaze.setText(_translate("HomePage", "Fog Dehaze"))
        self.pushButton.setText(_translate("HomePage", "Play / Stop"))
