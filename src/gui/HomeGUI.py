import json
import os
import sys

import torch

sys.path.append(os.getcwd())  # NOQA

from PyQt6.QtCore import QDateTime, QThread
from PyQt6.QtWidgets import QFileDialog, QMainWindow, QMessageBox

from src.core.ProcessVideoWorker import ProcessVideoWorker
from src.gui import gui_logger
from src.gui.MessageBox import MessageBox as msg
from src.view.home_page_ui import Ui_HomePage


class HomeGUI(QMainWindow):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.parent = parent
        self.home_ui = Ui_HomePage()
        self.home_ui.setupUi(self)

        self._load_configs()

        # Connect event
        self.home_ui.input_button.clicked.connect(self.choose_input_video)
        self.home_ui.output_button.clicked.connect(self.choose_export_folder)
        self.home_ui.process_button.clicked.connect(self.run_process_video)
        self.home_ui.conf_slide.valueChanged.connect(
            lambda value: self.home_ui.conf_label.setText(f"{round(value,2)} %")
        )
        self.home_ui.conf_slide.valueChanged

        # Thread
        self.process_video_thread = QThread()

    def _load_configs(self):
        with open("data/configs/system.json", "r", encoding="utf-8") as file:
            self.system_configs = json.load(file)
            gui_logger.info("System Configs Loaded")

        # Set the detect confidence for the slider
        self.home_ui.conf_slide.setValue(
            int(float(self.system_configs["detect_confidence"]) * 100)
        )
        self.home_ui.conf_label.setText(f"{self.home_ui.conf_slide.value()} %")

        # Set the device name
        self.home_ui.device_label.setText(self._get_device())

    def _get_open_dir(self) -> str:
        # Get the last visited folder from system configs
        last_visited_folder: str = self.system_configs["last_visited_folder"]

        # Check if the last visited folder exists
        if not (
            os.path.isdir(last_visited_folder) or os.path.exists(last_visited_folder)
        ):
            open_dir = os.getcwd()
        else:
            open_dir = last_visited_folder

        return open_dir

    def _get_device(self) -> str:
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
        else:
            device_name = "CPU"

        return f"Device: {device_name.upper()}"

    def update_log(self, text: str, color: str = "black"):
        # Get the current date and time in a user-friendly selfat
        current_datetime = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
        # Prepare the log entry with the selfatted date and time
        new_log = f"[{current_datetime}] {text}"
        # Apply appropriate HTML selfatting
        selfatted_log = f'<p style="color:{color}; margin: 0;">{new_log}</p>'
        # Append the selfatted log entry to the log output
        self.home_ui.log_area.setHtml(self.home_ui.log_area.toHtml() + selfatted_log)
        # Scroll to the bottom of the log output
        self.home_ui.log_area.verticalScrollBar().setValue(
            self.home_ui.log_area.verticalScrollBar().maximum()
        )

    def choose_input_video(self):
        open_dir = self._get_open_dir()

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Choose Video", open_dir, "Video Files (*.mp4 *.mkv *.avi)"
        )

        if file_path:
            self.home_ui.input_lineEdit.setText(file_path)
            self.system_configs["last_visited_folder"] = os.path.dirname(file_path)

            # Save the last visited folder to system configs
            with open("data/configs/system.json", "w", encoding="utf-8") as f:
                json.dump(self.system_configs, f, indent=4, ensure_ascii=False)

    def choose_export_folder(self):
        open_dir = self._get_open_dir()

        folder_path = QFileDialog.getExistingDirectory(
            parent=self, caption="Select Output Folder", directory=open_dir
        )

        if folder_path:
            # Update the last visited folder
            self.system_configs["last_visited_folder"] = folder_path

            # Save the system configs
            with open("data/configs/system.json", "w", encoding="utf-8") as f:
                json.dump(self.system_configs, f, indent=4, ensure_ascii=False)

            # Update the input_lineEdit
            self.home_ui.output_lineEdit.setText(folder_path)

    def _handle_success(self, output_video_path: str):
        msg.information_box_with_button(
            content=f"Video has successfully saved to path: {output_video_path}",
            button_name="Open Folder",
            button_action=lambda: os.startfile(os.path.dirname(output_video_path)),
        )

        # Stop the progress bar
        self.home_ui.progressBar.reset()

    def _set_up_progress_bar(self, total: int):
        self.home_ui.progressBar.setMaximum(total)

    def _increase_progress_bar(self):
        self.home_ui.progressBar.setValue(self.home_ui.progressBar.value() + 1)

    def run_process_video(self):
        input_video = self.home_ui.input_lineEdit.text()
        output_folder = self.home_ui.output_lineEdit.text()

        if not input_video or not output_folder:
            message = "Input Video or Output Folder is not provided"
            msg.warning_box(content=message)
            gui_logger.error(message)
            return

        user_choice = msg.yes_no_box(
            content="Are you sure you want to process the video?",
        )

        if user_choice == QMessageBox.StandardButton.No:
            return

        # Create worker
        self.process_video_worker = ProcessVideoWorker(
            video_path=input_video,
            save_folder=output_folder,
            sys_config=self.system_configs,
            dectect_conf=self.home_ui.conf_slide.value() / 100,
        )
        self.process_video_worker.moveToThread(self.process_video_thread)

        self.process_video_thread.started.connect(self.process_video_worker.run)

        self.process_video_worker.finished.connect(self.process_video_thread.quit)
        self.process_video_worker.finished.connect(
            self.process_video_worker.deleteLater
        )
        self.process_video_worker.finished.connect(self._handle_success)

        # Connect signals
        self.process_video_worker.logging.connect(self.update_log)
        self.process_video_worker.set_up_progress_bar.connect(self._set_up_progress_bar)
        self.process_video_worker.increase_progress_bar.connect(
            self._increase_progress_bar
        )

        # Start the thread
        self.process_video_thread.start()

    """Media Player"""

    def _start_player(self):
        pass
