import os
import sys

import numpy as np

sys.path.append(os.getcwd())  # NOQA

import uuid

import cv2
from PIL import Image
from PyQt6.QtCore import QDateTime, QObject, pyqtSignal

from src.core import core_logger
from src.models.DeepSort import DeepSort
from src.models.Models import Models
from src.models.YoloV8 import YoloV8


class ProcessVideoWorker(QObject):
    started = pyqtSignal()
    finished = pyqtSignal(str)
    logging = pyqtSignal(str, str)
    error = pyqtSignal(str)
    set_up_progress_bar = pyqtSignal(int)
    increase_progress_bar = pyqtSignal()

    def __init__(
        self,
        video_path: str,
        save_folder: str,
        sys_config: dict,
        dectect_conf: float = 0.4,
        options: dict = None,
        parent: QObject | None = ...,
    ) -> None:
        super(ProcessVideoWorker, self).__init__()
        self.video_path = video_path
        self.save_folder = save_folder
        self.sys_config = sys_config
        self.detect_conf = dectect_conf
        self.options = options
        core_logger.info(f"==>> detect_conf: {self.detect_conf}")

        self.detector = YoloV8(model_path=sys_config["yolo_model_path"])
        self.tracker = DeepSort(model_path=sys_config["deepsort_model_path"])
        self._classifiers = Models(model="resnet18", num_classes=3)
        self._classifiers.load_weight(sys_config["classifier_weight_path"])

    def draw_detection(
        self,
        img,
        bboxes,
        scores,
        class_ids,
        ids,
        classes=["xe so", "xe ga", "khong phai xe"],  # default classes
        mask_alpha=0.3,
    ):
        height, width = img.shape[:2]
        np.random.seed(0)

        colors = {0: [172, 47, 117], 1: [192, 67, 251], 2: [195, 103, 9]}

        core_logger.info(f"==>> colors: {colors}")

        # Create mask image
        mask_img = img.copy()
        det_img = img.copy()

        size = min([height, width]) * 0.0006
        text_thickness = int(min([height, width]) * 0.001)

        # Draw bounding boxes and labels of detections
        for bbox, score, class_id, id_ in zip(bboxes, scores, class_ids, ids):
            if class_id <= 2:
                color = colors[class_id]

                x1, y1, x2, y2 = bbox.astype(int)

                # Draw rectangle
                cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)

                # Draw fill rectangle in mask image
                cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)
                label = classes[class_id]
                caption = f"{label} ID: {id_}"
                (tw, th), _ = cv2.getTextSize(
                    text=caption,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=size,
                    thickness=text_thickness,
                )
                th = int(th * 1.2)

                cv2.rectangle(det_img, (x1, y1), (x1 + tw, y1 - th), color, -1)
                cv2.rectangle(mask_img, (x1, y1), (x1 + tw, y1 - th), color, -1)
                cv2.putText(
                    det_img,
                    caption,
                    (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    size,
                    (255, 255, 255),
                    text_thickness,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    mask_img,
                    caption,
                    (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    size,
                    (255, 255, 255),
                    text_thickness,
                    cv2.LINE_AA,
                )

            else:
                print("Class_id out of range colors")
                print(len(colors))
                print(class_id)
                continue

        return cv2.addWeighted(mask_img, mask_alpha, det_img, 1 - mask_alpha, 0)

    def _classify(self, bboxes, frame):
        # Convert frame to PIL image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)

        class_ids = []
        for bbox in bboxes:
            x, y, w, h = bbox
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)

            core_logger.info(f"Coordinates: {x1, y1, x2, y2}")
            cropped_img = img.crop((x1, y1, x2, y2))
            class_id = self._classifiers.infer(cropped_img)

            unique_name = str(uuid.uuid4())

            # Save the cropped result to check
            cropped_img.save(f".temp/{unique_name}_{class_id}.jpg")

            core_logger.info(f"Class ID: {class_id}")
            class_ids.append(class_id)

        return class_ids

    def run(self):
        self.started.emit()

        ############# Define the video capture object and its properties
        cap = cv2.VideoCapture(self.video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")

        ############# Define the video writer object
        datetime_now = QDateTime.currentDateTime().toString("yyyy-MM-dd_hh-mm-ss")
        video_path = os.path.join(self.save_folder, f"{datetime_now}.avi")
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        ############# Some variables for tracking
        all_tracking_results = []
        tracked_ids = np.array([], dtype=np.int32)

        ############# Start processing the video into frames
        self.set_up_progress_bar.emit(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.logging.emit(
            f"Processing video: {os.path.basename(self.video_path)}", "blue"
        )
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.increase_progress_bar.emit()
            self.logging.emit(
                f"Processing frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}/{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}",
                "black",
            )
            core_logger.info(
                f"Processing frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}/{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}"
            )

            ############# Enhance the frame (lighting or contrast, etc.)

            if self.options.get("light_enhance") is True:
                # TODO: Implement the light enhancement code here
                pass

            if self.options.get("fog_dehaze") is True:
                # TODO: Implement the fog dehaze code here
                pass

            ############# Detect objects in the frame
            detector_results = self.detector.detect(conf=self.detect_conf, frame=frame)
            bboxes, scores, class_ids = detector_results

            ############# Classify detected objects
            new_class_ids = self._classify(bboxes, frame)

            ############# Track detected objects
            tracker_pred = self.tracker.tracking(
                origin_frame=frame,
                bboxes=bboxes,
                scores=scores,
                class_ids=new_class_ids,
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

                result_img = self.draw_detection(
                    img=frame,
                    bboxes=bboxes,
                    scores=conf_scores,
                    class_ids=class_ids,
                    ids=track_ids,
                )
            else:
                result_img = frame

            ############# Write the frame to the video
            all_tracking_results.append(tracker_pred)

            out.write(result_img)

        cap.release()
        out.release()

        self.logging.emit(f"Video has been saved to {video_path}", "green")
        self.finished.emit(video_path)


if __name__ == "__main__":
    worker = ProcessVideoWorker("assets/test_vid.mp4", "assets")
    worker.run()
