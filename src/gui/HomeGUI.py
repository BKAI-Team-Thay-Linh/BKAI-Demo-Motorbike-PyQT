import json
import os
import sys

import torch

sys.path.append(os.getcwd())  # NOQA

from PyQt6.QtCore import QDateTime, QThread, QUrl
from PyQt6.QtGui import QShortcut
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import QFileDialog, QMainWindow, QMessageBox, QStyle

from src.core import core_logger
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
        self._init_media_player()

        # Connect event
        self.home_ui.input_button.clicked.connect(self.choose_input_video)
        self.home_ui.process_button.clicked.connect(self.run_process_video)
        self.home_ui.conf_slide.valueChanged.connect(
            lambda value: self.home_ui.detect_conf_label.setText(f"{round(value,2)} %")
        )

        # Thread
        self.process_video_thread = QThread()

    """Utils functions"""

    def _init_media_player(self):
        self.media_player = QMediaPlayer(None)
        self.video_widget = QVideoWidget(parent=self.home_ui.media_widget)

        self.video_widget.setFixedSize(self.home_ui.media_widget.size())

        self.media_player.setVideoOutput(self.video_widget)
        self.video_widget.show()

        self.home_ui.play_video_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        )

        self.home_ui.horizontalSlider.setRange(0, 0)

        # Add shortcut for full screen
        QShortcut("F", self.video_widget, self._full_screen_event)
        QShortcut("Space", self.video_widget, self._play_video)

    def _load_configs(self):
        with open("data/configs/system.json", "r", encoding="utf-8") as file:
            self.system_configs = json.load(file)
            gui_logger.info("System Configs Loaded")

        # Set the detect confidence for the slider
        self.home_ui.conf_slide.setValue(
            int(float(self.system_configs["detect_confidence"]) * 100)
        )
        self.home_ui.detect_conf_label.setText(f"{self.home_ui.conf_slide.value()} %")

        # Set the device name
        self.home_ui.device_label.setText(self._get_device())

        # Load the weights
        self._add_detection_weights()
        self._add_classification_models()

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
            self.device = "cuda"
            device_name = torch.cuda.get_device_name(0)
        else:
            self.device = "cpu"
            device_name = "CPU"

        return f"Device: {device_name.upper()}"

    def _update_log(self, text: str, color: str = "black"):
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

    def _handle_success(self, summary_report: str):
        video_name = os.listdir(".temp/output_video")[0]

        msg.information_box_with_button(
            content=summary_report,
            button_name="Preview",
            button_action=lambda: self._start_player(
                f".temp/output_video/{video_name}"
            ),
        )

        # Stop the progress bar
        self.home_ui.progressBar.setValue(0)
        self.home_ui.progressBar.reset()

    def _set_up_progress_bar(self, total: int):
        self.home_ui.progressBar.setMaximum(total)

    def _increase_progress_bar(self):
        self.home_ui.progressBar.setValue(self.home_ui.progressBar.value() + 1)

    """I/O functions"""

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

    """Media Player"""

    def _full_screen_event(self):
        if self.video_widget.isFullScreen():
            self.video_widget.setFullScreen(False)
        else:
            self.video_widget.setFullScreen(True)

    def _media_playback_duration_changed(self, duration):
        self.max_video_duration = duration
        self.home_ui.duration_label.setText(f"0:/{duration}s")
        self.home_ui.horizontalSlider.setRange(0, duration)

    def _media_playback_position_changed(self, position):
        self.home_ui.duration_label.setText(
            f"{round(position/1000, 2)}/{round(self.max_video_duration/1000, 2)}s"
        )
        self.home_ui.horizontalSlider.setValue(position)

    def _media_playback_state_changed(self, state):
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.home_ui.play_video_button.setIcon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause)
            )
        else:
            self.home_ui.play_video_button.setIcon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
            )

    def _play_video(self):
        print(self.media_player.playbackState())
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PausedState:
            self.media_player.play()
            core_logger.info("Video Played")
        else:
            self.media_player.pause()
            core_logger.info("Video Paused")

    def _start_player(self, video_path: str):
        # Play the video in the media widget
        self.media_player.setSource(QUrl.fromLocalFile(video_path))
        self.home_ui.play_video_button.clicked.connect(self._play_video)
        self.home_ui.horizontalSlider.sliderMoved.connect(self.media_player.setPosition)
        self.media_player.durationChanged.connect(self._media_playback_duration_changed)
        self.media_player.positionChanged.connect(self._media_playback_position_changed)
        self.media_player.playbackStateChanged.connect(
            self._media_playback_state_changed
        )

        self.home_ui.zoom_button.clicked.connect(self._full_screen_event)

        # Play the video
        self.media_player.play()

    """Models Choosing Functions"""

    def _add_detection_weights(self):
        yolo_weights_dir = "weight/yolo"
        self.home_ui.yolo_combobox.addItems(os.listdir(yolo_weights_dir))

    def _add_classification_models(self):
        classification_weights_dir = "weight/classify"
        models = map(lambda x: x.split(".")[0], os.listdir(classification_weights_dir))
        self.home_ui.classify_combobox.addItems(models)

    """Processing Functions"""

    def run_process_video(self):
        input_video = self.home_ui.input_lineEdit.text()

        if not input_video:
            message = "Input Video is not provided"
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
            sys_config=self.system_configs,
            device=self.device,
            detection_weight_path=f"weight/yolo/{self.home_ui.yolo_combobox.currentText()}",
            classification_model=self.home_ui.classify_combobox.currentText(),
            dectect_conf=self.home_ui.conf_slide.value() / 100,
            options={
                "light_enhance": True
                if self.home_ui.opt_light_enhance.isChecked()
                else False,
                "fog_dehaze": True
                if self.home_ui.opt_fog_dehaze.isChecked()
                else False,
            },
        )
        self.process_video_worker.moveToThread(self.process_video_thread)

        self.process_video_thread.started.connect(self.process_video_worker.run)

        self.process_video_worker.finished.connect(self.process_video_thread.quit)
        self.process_video_worker.finished.connect(
            self.process_video_worker.deleteLater
        )
        self.process_video_worker.finished.connect(self._handle_success)

        # Connect signals
        self.process_video_worker.logging.connect(self._update_log)
        self.process_video_worker.set_up_progress_bar.connect(self._set_up_progress_bar)
        self.process_video_worker.increase_progress_bar.connect(
            self._increase_progress_bar
        )

        # Start the thread
        self.process_video_thread.start()
