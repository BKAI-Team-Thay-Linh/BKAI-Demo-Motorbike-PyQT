import json
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.append(os.getcwd())

import time
from concurrent.futures import ThreadPoolExecutor, as_completed  # NOQA

import cv2
from PIL import Image
from PyQt6.QtCore import QObject, pyqtSignal

import src.utils.constants as const
from src.core import core_logger
from src.models.classify.main import Models
from src.models.detector import YoloDectector
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
            will_classify=False,
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
            # self.logging.emit(f"Detecting objects in the frame {image_name}", "black")
            print(f"Detecting objects in the frame {image_name}", end="\r")
            self.increase_progress_bar.emit()

            # Read the frame
            image_path = os.path.join(self.TEMP_FOLDER_EXTRACT, image_name)
            frame = cv2.imread(image_path)
            bboxes, scores = self.detector.detect(conf=self.detect_conf, frame=frame)
            output.append((idx, bboxes, scores, [], frame))
        print()
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
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {
                executor.submit(self.__classify, bboxes, frame): (bboxes, frame)
                for idx, (bboxes, _, _, frame) in enumerate(output)
            }

            self.set_up_progress_bar.emit(len(futures))
            for idx, future in enumerate(as_completed(futures)):
                self.increase_progress_bar.emit()
                print(f"Classifying frame {idx} / {len(futures)}")
                bboxes, frame = futures[future]
                class_ids.append(future.result())

            self.set_up_progress_bar.emit(0)

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

    def _write_frames_to_video_ffmpeg(self, output_frames: list[np.ndarray]) -> str:
        core_logger.info("Writing the frames to the video using FFmpeg ...")
        self.logging.emit("Writing the frames to the video using FFmpeg ...", "blue")

        # Save all the frames into a folder
        os.makedirs(".temp/output_frames", exist_ok=True)

        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {
                executor.submit(
                    cv2.imwrite,
                    f".temp/output_frames/image_{str(idx).zfill(8)}.jpg",
                    frame,
                ): f"image_{str(idx).zfill(8)}.jpg"
                for idx, frame in enumerate(output_frames)
            }

            self.set_up_progress_bar.emit(len(futures))
            for idx, future in enumerate(as_completed(futures)):
                self.increase_progress_bar.emit()
                print(f"Writing frame {idx} / {len(futures)}")
                future.result()

    def _merge_frames_to_video_ffmpeg(self) -> str:
        # Write the frames to the video using FFmpeg
        if const.PLATFORM == "WIN":
            ffmpeg_path = os.path.join(
                os.path.realpath("ffmpeg/bin/ffmpeg.exe"),
            )
        else:
            ffmpeg_path = "ffmpeg"

        shutil.rmtree(".temp/output_video", ignore_errors=True)
        os.makedirs(".temp/output_video", exist_ok=True)

        frames_path = os.path.realpath(".temp/output_frames")
        output_video_name = (
            os.path.basename(self.video_path).split(".")[0] + "_processed.mp4"
        )
        output_video_path = os.path.realpath(f".temp/output_video/{output_video_name}")

        if const.PLATFORM == "WIN":
            self.split_process = subprocess.Popen(
                f"{ffmpeg_path} -i {frames_path}\image_%08d.jpg  -threads {self.sys_config['threads']} -r {self.fps} -c:v libx264 -pix_fmt yuv420p {output_video_path}",
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )
        else:
            self.split_process = subprocess.Popen(
                f"{ffmpeg_path} -i {frames_path}/image_%08d.jpg  -threads {self.sys_config['threads']} -r {self.fps} -c:v libx264 -pix_fmt yuv420p {output_video_path}",
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )

        counter = 1
        while self.split_process.poll() is None:
            if counter > 3:
                counter = 1
                print()
            print(f"Waiting for the split process to finish {'. ' * counter}", end="\r")
            time.sleep(1)
            counter += 1

    def run(self):
        self.started.emit()
        shutil.rmtree(".temp", ignore_errors=True)
        os.makedirs(".temp", exist_ok=True)

        start_time = time.time()
        ############# Define the video capture object and its properties #############
        cap = cv2.VideoCapture(self.video_path)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))

        ############# Split the video into frames #############
        self._split_video_into_frames(video_path=self.video_path, fps=self.fps)

        ############# Detect bboxes #############
        detect_start_time = time.time()
        output = self._detect_bboxes()
        detect_elapsed_time = time.time() - detect_start_time

        ############# Classify detected objects #############
        classifying_start_time = time.time()
        new_class_ids = self._classify_frames(output)
        classify_elapsed_time = time.time() - classifying_start_time

        ############# Track detected objects #############
        tracking_start_time = time.time()
        output_frames = self._track_objects(output, new_class_ids)
        tracking_elapsed_time = time.time() - tracking_start_time

        ############# Write the frames to temp folder #############
        self._write_frames_to_video_ffmpeg(output_frames)

        # Release memory
        cap.release()
        del cap
        del output
        del new_class_ids
        del output_frames

        ############# Merge the frames to video using FFmpeg #############
        self._merge_frames_to_video_ffmpeg()

        total_elapsed_time = time.time() - start_time
        self.logging.emit("Processing video has finished", "green")

        summary_report = f"""
Summary Report:
    - Detecting objects: {detect_elapsed_time:.2f}s
    - Classifying objects: {classify_elapsed_time:.2f}s
    - Tracking objects: {tracking_elapsed_time:.2f}s
    - Total time elapsed: {total_elapsed_time:.2f}s
        """

        self.finished.emit(summary_report)


if __name__ == "__main__":
    start_time = time.time()

    worker = ProcessVideoWorker(
        video_path="assets/5min.mp4",
        sys_config=json.load(open("data/configs/system.json", "r", encoding="utf-8")),
        options={"light_enhance": False, "fog_dehaze": False},
        device="cuda",
        detection_weight_path="weight/yolo/yolov9_best.pt",
        classification_model="ResNet18",
    )

    worker.run()

    print(f"\nTime elapsed: {time.time() - start_time:.2f}s")
