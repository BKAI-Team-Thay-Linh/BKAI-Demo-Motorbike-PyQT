import os
import sys
sys.path.append(os.getcwd())  # NOQA

from PyQt6.QtWidgets import QMessageBox, QPushButton


class MessageBox():

    @staticmethod
    def information_box(content: str):
        message_box = QMessageBox()
        message_box.setIcon(QMessageBox.Icon.Information)
        # message_box.setWindowIcon(QIcon(":/icons/folinas.ico"))
        message_box.setWindowTitle("Information")
        message_box.setText(content)
        message_box.exec()

    @staticmethod
    def warning_box(content: str):
        message_box = QMessageBox()
        message_box.setIcon(QMessageBox.Icon.Warning)
        message_box.setWindowTitle("Warning")
        # message_box.setWindowIcon(QIcon(":/icons/folinas.ico"))
        message_box.setText(content)
        message_box.exec()

    @staticmethod
    def warning_box_with_button(content: str, button_name: str, button_action: callable):  # type: ignore
        message_box = QMessageBox()
        message_box.setIcon(QMessageBox.Icon.Warning)
        message_box.setWindowTitle("Warning")
        # message_box.setWindowIcon(QIcon(":/icons/folinas.ico"))
        message_box.setText(content)
        # This button can be set action and name by the caller
        button: QPushButton = message_box.addButton(button_name, QMessageBox.ButtonRole.YesRole)  # type: ignore
        button.setFixedWidth(150)
        button.move(400, 200)
        # If the button is clicked, the action will be executed
        button.clicked.connect(lambda: button_action())
        message_box.buttonClicked.connect(lambda: button_action())
        message_box.exec()

    @staticmethod
    def information_box_with_button(content: str, button_name: str, button_action: callable):  # type: ignore
        message_box = QMessageBox()
        message_box.setIcon(QMessageBox.Icon.Information)  # type: ignore
        message_box.setWindowTitle("Information")
        # message_box.setWindowIcon(QIcon(":/icons/folinas.ico"))
        message_box.setText(content)
        # This button can be set action and name by the caller
        button: QPushButton = message_box.addButton(button_name, QMessageBox.ButtonRole.YesRole)  # type: ignore
        button.setFixedWidth(150)
        button.move(400, 200)
        # If the button is clicked, the action will be executed
        message_box.buttonClicked.connect(lambda: button_action())
        message_box.exec()

    @staticmethod
    def critical_box(content: str):
        message_box = QMessageBox()
        message_box.setIcon(QMessageBox.Icon.Critical)
        message_box.setWindowTitle("Error")
        # message_box.setWindowIcon(QIcon(":/icons/folinas.ico"))
        message_box.setText(content)
        message_box.exec()

    @staticmethod
    def yes_no_box(content: str):
        message_box = QMessageBox()
        message_box.setIcon(QMessageBox.Icon.Question)
        message_box.setWindowTitle("Question")
        # message_box.setWindowIcon(QIcon(":/icons/folinas.ico"))
        message_box.setText(content)
        message_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        result = message_box.exec()
        return result
