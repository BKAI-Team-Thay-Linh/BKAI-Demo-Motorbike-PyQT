import os
import sys
from types import TracebackType

from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *

from src import src_logger
from src.gui.HomeGUI import *


def critical_error(message: str):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Icon.Critical)
    msg.setText("An error has occurred:")
    msg.setInformativeText(f"\n{message}")
    msg.setWindowTitle("Critical Error")
    msg.setStandardButtons(QMessageBox.StandardButton.Ok)
    msg.exec()


def handle_exception(exctype, value, tb: TracebackType):
    src_logger.error('Uncaught Exception', exc_info=(exctype, value, tb))
    sys.__excepthook__(exctype, value, tb)
    critical_error(f'An error has occurred: \n\n{value}. For more information, please see the log file.')


sys.excepthook = handle_exception


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = QMainWindow()
    main_window.show()
    sys.exit(app.exec())
