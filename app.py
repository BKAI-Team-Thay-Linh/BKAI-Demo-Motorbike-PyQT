import sys
import traceback
from types import TracebackType

from PyQt6.QtWidgets import QApplication, QMessageBox

from src import src_logger
from src.gui.HomeGUI import HomeGUI


def critical_error(message: str):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Icon.Critical)
    msg.setText("An error has occurred:")
    msg.setInformativeText(f"\n{message}")
    msg.setWindowTitle("Critical Error")
    msg.setStandardButtons(QMessageBox.StandardButton.Ok)
    msg.exec()


def handle_exception(exctype, value, tb: TracebackType):
    src_logger.error("Uncaught Exception", exc_info=(exctype, value, tb))
    sys.__excepthook__(exctype, value, tb)
    critical_error(f"{value}.\n\nFor more information, please see the log file")


if __name__ == "__main__":
    sys.excepthook = handle_exception
    app = QApplication(sys.argv)

    try:
        # Create the main window
        window = HomeGUI()
        window.show()

        # Run the application
        sys.exit(app.exec())
    except Exception as e:
        print(traceback.format_exc())
        critical_error(str(e))
        sys.exit(1)
