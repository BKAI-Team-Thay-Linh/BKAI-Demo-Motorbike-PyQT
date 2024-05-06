import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

import numpy as np

sys.path.append(os.getcwd())

import time
from concurrent.futures import ThreadPoolExecutor, as_completed  # NOQA

import cv2
from PIL import Image
from PyQt6.QtCore import QObject, pyqtSignal

import src.utils.constants as const
from src.core import core_logger
from src.models.detector import YoloDectector
from src.models.Models import Models
from src.models.tracker import DeepSort
from src.utils.draw import draw_bboxes


class ProcessVideoWorker(QObject):
    started = pyqtSignal()
    finished = pyqtSignal(str)
    logging = pyqtSignal(str, str)
    error = pyqtSignal(str)
    set_up_progress_bar = pyqtSignal(int)
    increase_progress_bar = pyqtSignal()

    TEMP_FOLDER_EXTRACT = ".temp/extracted_frames"
    TEMP_FOLDER_SAVE_VIDEO = ".temp/processed_video"

    def __init__(
        self,
        video_path: str = ...,
        sys_config: dict = ...,
        device: str = "cpu",
        detection_weight_path: str = ...,
        classification_model: str = ...,
        dectect_conf: float = 0.4,
        options: dict = ...,
        parent: QObject | None = ...,
    ) -> None:
        super(ProcessVideoWorker, self).__init__()

        # Define the attributes
        self.video_path = video_path
        self.sys_config = sys_config
        self.device = device
        self.detection_weight_path = detection_weight_path
        self.classification_model = classification_model
        self.detect_conf = dectect_conf
        self.options = options
        self.parent = parent

        # Init the models
        self.detector = YoloDectector(
            model_path=self.detection_weight_path,
            device=self.device,
        )
        self.tracker = DeepSort(
            model_path=sys_config["deepsort_model_path"],
        )
        self.classifier = Models(
            model=self.classification_model.lower(),
            num_classes=3,
        )

        weight_path = None
        for name in os.listdir("weight/classify"):
            if name.startswith(self.classification_model):
                weight_path = f"weight/classify/{name}"
                break
        if weight_path is None:
            raise FileNotFoundError(
                f"Weight file for {self.classification_model} not found"
            )

        self.classifier.load_weight(weight_path)

    def __classify(self, bboxes, frame) -> list[int]:
        # Convert frame to PIL image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)

        class_ids = []
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            # print(x1, y1, x2, y2)

            # Crop the frame
            cropped_img = img.crop((x1, y1, x2, y2))
            class_id = self.classifier.infer(cropped_img)

            # Save the cropped result to check
            # unique_name = str(uuid.uuid4())
            # os.makedirs(".temp/predicted_frames", exist_ok=True)
            # cropped_img.save(f".temp/predicted_frames/{unique_name}_{class_id}.jpg")

            class_ids.append(class_id)

        return class_ids

    """Steps to process the video"""

    def _split_video_into_frames(self, video_path: str, fps: int) -> None:
        core_logger.info("Splitting video into frames using FFmpeg ...")
        self.logging.emit("Splitting video into frames using FFmpeg ...", "blue")

        if const.PLATFORM == "WIN":
            ffmpeg_path = os.path.join(
                os.path.realpath("ffmpeg/bin/ffmpeg.exe"),
            )
        else:
            ffmpeg_path = "ffmpeg"

        shutil.rmtree(self.TEMP_FOLDER_EXTRACT, ignore_errors=True)
        os.makedirs(self.TEMP_FOLDER_EXTRACT, exist_ok=True)

        if const.PLATFORM == "WIN":
            self.split_process = subprocess.Popen(
                f"{ffmpeg_path} -i {os.path.realpath(video_path)} -threads {self.sys_config['threads']} -r {fps} -q:v 2 {os.path.realpath(self.TEMP_FOLDER_EXTRACT)}\image_%08d.jpg",
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )
        else:
            self.split_process = subprocess.Popen(
                f"{ffmpeg_path} -i {os.path.realpath(video_path)} -threads {self.sys_config['threads']} -r {fps} -q:v 2 {os.path.realpath(self.TEMP_FOLDER_EXTRACT)}/image_%08d.jpg",
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )

    def _preprocess_frame(self, frame) -> np.ndarray:
        if self.options.get("light_enhance") is True:
            out_frame = self.light_enhance(frame)
        else:
            out_frame = frame

        if self.options.get("fog_dehaze") is True:
            out_frame = self.fog_dehaze(out_frame)
        else:
            out_frame = frame

        return out_frame

    def _detect_bboxes_in_frame(self, image_name: str) -> None:
        print(f"Detecting objects in the frame {image_name}")

        # Read the frame
        image_path = os.path.join(self.TEMP_FOLDER_EXTRACT, image_name)
        frame = cv2.imread(image_path)
        bboxes, scores, class_ids = self.detector.detect(
            conf=self.detect_conf, frame=frame
        )

        return bboxes, scores, class_ids, frame

    def _detect_bboxes(self) -> list[tuple]:
        core_logger.info("Detecting objects in the frame ...")
        self.logging.emit("Detecting objects in the frame ...", "blue")

        poll = self.split_process.poll()

        sec = 0
        while poll is None:
            print(
                f"Waiting for the split process to finish. Time elapse: {sec}s",
                end="\r",
            )
            sec += 1
            poll = self.split_process.poll()
            time.sleep(1)

        print("\nSplit process has finished. Now detecting objects ...")

        output = []
        self.set_up_progress_bar.emit(len(os.listdir(self.TEMP_FOLDER_EXTRACT)))
        for idx, image_name in enumerate(os.listdir(self.TEMP_FOLDER_EXTRACT)):
            self.logging.emit(f"Detecting objects in the frame {image_name}", "black")
            print(f"Detecting objects in the frame {image_name}")
            self.increase_progress_bar.emit()

            # Read the frame
            image_path = os.path.join(self.TEMP_FOLDER_EXTRACT, image_name)
            frame = cv2.imread(image_path)
            bboxes, scores, class_ids = self.detector.detect(
                conf=self.detect_conf, frame=frame
            )
            output.append((idx, bboxes, scores, class_ids, frame))
        self.set_up_progress_bar.emit(0)

        output.sort(key=lambda x: x[0])
        output = [x[1:] for x in output]

        return output

    # Classify detected objects
    def _classify_frames(self, output: list) -> list[int]:
        core_logger.info("Classifying detected objects ...")
        self.logging.emit("Classifying detected objects ...", "blue")

        class_ids = []

        # Try with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self.__classify, bboxes, frame): (bboxes, frame)
                for idx, (bboxes, _, _, frame) in enumerate(output)
            }

            for idx, future in enumerate(as_completed(futures)):
                print(f"Classifying frame {idx} / {len(futures)}")
                bboxes, frame = futures[future]
                class_ids.append(future.result())

        print(class_ids[:5])

        return class_ids

    # Track detected objects
    def _track_objects(
        self, output: list, new_class_ids: list[int]
    ) -> list[np.ndarray]:
        core_logger.info("Tracking detected objects ...")
        self.logging.emit("Tracking detected objects ...", "blue")

        output_frames = []
        tracked_ids = np.array([], dtype=np.int32)

        # Reset progress bar
        self.set_up_progress_bar.emit(len(output))
        for (bboxes, scores, _, frame), class_ids in zip(output, new_class_ids):
            self.increase_progress_bar.emit()
            tracker_pred = self.tracker.tracking(
                origin_frame=frame,
                bboxes=bboxes,
                scores=scores,
                class_ids=class_ids,
            )

            if tracker_pred.size > 0:
                bboxes = tracker_pred[:, :4]
                class_ids = tracker_pred[:, 4].astype(int)
                conf_scores = tracker_pred[:, 5]
                track_ids = tracker_pred[:, 6].astype(int)

                # Get new tracking IDs
                new_ids = np.setdiff1d(track_ids, tracked_ids)

                # Store new tracking IDs
                tracked_ids = np.concatenate((tracked_ids, new_ids))

                result_img = draw_bboxes(
                    img=frame,
                    bboxes=bboxes,
                    scores=conf_scores,
                    class_ids=class_ids,
                    track_ids=track_ids,
                )
            else:
                result_img = frame

            output_frames.append(result_img)
        self.set_up_progress_bar.emit(0)

        return output_frames

    def _write_frames_to_video(self, output_frames: list[np.ndarray]) -> str:
        core_logger.info("Writing the frames to the video ...")
        self.logging.emit("Writing the frames to the video ...", "blue")

        fourcc = cv2.VideoWriter_fourcc(*"H264")

        old_video_name = os.path.basename(self.video_path)
        time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_video_name = old_video_name.split(".")[0] + f"_processed_{time_now}.mp4"

        # Temporary save it to TEMP_FOLDER_SAVE_VIDEO
        os.makedirs(self.TEMP_FOLDER_SAVE_VIDEO, exist_ok=True)
        out_video_path = os.path.join(self.TEMP_FOLDER_SAVE_VIDEO, out_video_name)

        out = cv2.VideoWriter(
            out_video_path, fourcc, self.fps, (self.width, self.height)
        )

        self.set_up_progress_bar.emit(len(output_frames))
        for frame in output_frames:
            self.increase_progress_bar.emit()
            out.write(frame)
            cv2.imshow("Frame", frame)
            c = cv2.waitKey(1)
            if c & 0xFF == ord("q"):
                break
        self.set_up_progress_bar.emit(0)
        out.release()

        return out_video_path

    def run(self):
        self.started.emit()
        shutil.rmtree(".temp", ignore_errors=True)
        os.makedirs(".temp", exist_ok=True)

        ############# Define the video capture object and its properties #############
        cap = cv2.VideoCapture(self.video_path)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))

        ############# Split the video into frames #############
        self._split_video_into_frames(video_path=self.video_path, fps=self.fps)

        ############# Detect bboxes #############
        output = self._detect_bboxes()

        ############# Classify detected objects #############
        new_class_ids = self._classify_frames(output)

        ############# Track detected objects #############
        output_frames = self._track_objects(output, new_class_ids)

        ############# Write the frames to the video #############
        out_path = self._write_frames_to_video(output_frames)

        self.logging.emit("Processing video has finished", "green")

        self.finished.emit(out_path)


if __name__ == "__main__":
    worker = ProcessVideoWorker(
        video_path="assets/hoangcau.mp4",
        sys_config=json.load(open("data/configs/system.json", "r", encoding="utf-8")),
        options={"light_enhance": False, "fog_dehaze": False},
        device="cuda",
        detection_weight_path="weight/yolo/yolov9_best.pt",
        classification_model="ResNet18",
    )

    worker.run()
