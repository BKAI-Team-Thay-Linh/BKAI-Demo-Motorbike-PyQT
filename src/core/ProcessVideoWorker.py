import os
import shutil
import sys
import numpy as np
import torch
sys.path.append(os.getcwd())  # NOQA

import cv2
from PIL import Image

from PyQt6.QtCore import *
from src.core import core_logger
from src.models.Models import Models
import ultralytics as ult


class ProcessVideoWorker(QObject):

    started = pyqtSignal()
    finished = pyqtSignal()
    logging = pyqtSignal(str)
    error = pyqtSignal(str)
    set_up_progress_bar = pyqtSignal(int)
    increase_progress_bar = pyqtSignal()

    def __init__(self, video_path: str, save_folder: str, parent: QObject | None = ...) -> None:
        super(ProcessVideoWorker, self).__init__()
        self.video_path = video_path
        self.save_folder = save_folder

        # Parameters
        self.bbox_coordinates = {}  # For drawing the bounding box on the original frame

        # Init the YOLO model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.yolo = ult.YOLO('weight/yolov8m.pt').to(device=device)
        core_logger.info('YOLO model has been loaded')

        self.classification_model = Models(model='resnet18', num_classes=3)
        self.classification_model.load_weight('weight/resnet18.ckpt')
        core_logger.info('Classification model has been loaded')

    def extract_video_into_frames(self):
        """
            Step 1:
                Extract video into frames and save them in the .temp folder
        """
        os.makedirs('.temp/extracted_frame', exist_ok=True)

        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.set_up_progress_bar.emit(total_frames)

        core_logger.info(f'Extracting video into frames, total frames: {total_frames}')
        self.logging.emit(f'Extracting video into frames, total frames: {total_frames}')

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Get the frame id and name then save path
                frame_id = int(cap.get(1))
                frame_name = f'{frame_id}.jpg'
                frame_path = os.path.join('.temp', 'extracted_frame', frame_name)

                self.increase_progress_bar.emit()
                core_logger.info(f'Extracting Frame: {frame_id}/{total_frames}')

                # Save the frame
                cv2.imwrite(frame_path, frame)
            else:
                break

        cap.release()
        core_logger.info('Video Extraction Complete')

        # Reset the progress bar
        self.set_up_progress_bar.emit(0)

    def _crop_bbox_and_save(self, frame_path: str, conf: float = 0.4):
        """
            Step 2:
                Crop the frame into bounding box
        """
        os.makedirs('.temp/cropped_bboxes', exist_ok=True)

        # Open the frame
        original_frame = Image.open(frame_path)
        original_frame_name = os.path.basename(frame_path).split('.')[0]
        bboxes_info: list = []

        # Get the bounding box
        core_logger.info(f'Detecting bboxes in the frame: {original_frame_name}')
        detection_result = self.yolo.predict(original_frame, conf=conf, classes=3)
        bounding_boxes_coor = detection_result[0].boxes.xywh.cpu().numpy().tolist()
        print(bounding_boxes_coor)

        for idx, bb in enumerate(bounding_boxes_coor):
            x, y, w, h = bb
            bbox_name = f'{original_frame_name}_bbox_{idx}.jpg'
            core_logger.info(f'Extracting BBox: {bbox_name}')

            # Convert to the coordinates in the original image
            x_min = int(round(x - (w / 2)))
            y_min = int(round(y - (h / 2)))
            x_max = x_min + int(round(w))
            y_max = y_min + int(round(h))

            bboxes_info.append({
                'name': bbox_name,
                'coor': (x_min, y_min, x_max, y_max),
                'class': -1  # -1 means not classified yet
            })

            # Crop the image
            crop_image = original_frame.crop((x_min, y_min, x_max, y_max))
            crop_image.save(os.path.join('.temp', 'cropped_bboxes', bbox_name))

        self.bbox_coordinates[original_frame_name + '.jpg'] = bboxes_info
        core_logger.info(f'Frame: {original_frame_name} has been processed')

    def _draw_bb(self, thickness: int, font_scale: float, original_image: Image, bb_boxes: list, classes: list):
        # Convert the image to numpy array
        original_image = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)

        for bb, cls in zip(bb_boxes, classes):
            x_min, y_min, x_max, y_max = bb

            # Class 0 will be in blue, class 1 will be in green, else transparent
            if cls not in (0, 1):
                continue

            if cls == 0:
                color = (255, 0, 0)
                label = 'xe so'
            elif cls == 1:
                color = (0, 255, 0)
                label = 'xe ga'

            # Draw the bounding box
            cv2.rectangle(original_image, (x_min, y_min), (x_max, y_max), color, thickness)

            # Draw the label with the bounding box
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_min = max(y_min, label_size[1])
            cv2.rectangle(original_image, (x_min, y_min - label_size[1]),
                          (x_min + label_size[0], y_min + base_line),
                          color, cv2.FILLED)
            cv2.putText(original_image, label, (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)

        return Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))

    def run(self):
        self.started.emit()

        # Step 1: Extract video into frames
        self.extract_video_into_frames()

        # Step 2: Crop the frame into bounding box
        self.set_up_progress_bar.emit(len(os.listdir('.temp/extracted_frame')))
        self.logging.emit('Cropping the frame into bounding box...')

        for frame in os.listdir('.temp/extracted_frame'):
            self.increase_progress_bar.emit()
            frame_path = os.path.join('.temp', 'extracted_frame', frame)
            self._crop_bbox_and_save(frame_path)

        # Reset the progress bar
        self.set_up_progress_bar.emit(0)

        # Step 3: Run the classification model on the cropped bboxes
        core_logger.info('Running the classification model on the cropped bboxes...')
        self.logging.emit('Running the classification model on the cropped bboxes...')

        self.set_up_progress_bar.emit(len(os.listdir('.temp/cropped_bboxes')))

        for frame in self.bbox_coordinates.keys():
            for bbox in self.bbox_coordinates[frame]:
                self.increase_progress_bar.emit()

                bbox_path = os.path.join('.temp', 'cropped_bboxes', bbox['name'])
                bbox_image = Image.open(bbox_path)
                bbox['class'] = self.classification_model.infer(bbox_image)
                core_logger.info(f'Classified {bbox["name"]} as {bbox["class"]}')

        # Reset the progress bar
        self.set_up_progress_bar.emit(0)

        # Step 4: Draw the bounding box on the original frame
        core_logger.info('Drawing the bounding box on the original frame...')
        self.logging.emit('Drawing the bounding box on the original frame...')

        os.makedirs('.temp/annotated_frame', exist_ok=True)

        self.set_up_progress_bar.emit(len(self.bbox_coordinates))

        for frame in sorted(self.bbox_coordinates.keys()):
            self.increase_progress_bar.emit()
            frame_path = os.path.join('.temp', 'extracted_frame', frame)
            frame_image = Image.open(frame_path)
            bboxes_info = self.bbox_coordinates[frame]

            bb_boxes = [bbox['coor'] for bbox in bboxes_info]
            classes = [bbox['class'] for bbox in bboxes_info]

            new_frame = self._draw_bb(2, 0.5, frame_image, bb_boxes, classes)
            core_logger.info(f'Annotated Frame: {frame}')
            new_frame.save(os.path.join('.temp', 'annotated_frame', frame))

        # Reset the progress bar
        self.set_up_progress_bar.emit(0)

        # Step 5: Save the annotated frames into a video
        core_logger.info('Saving the annotated frames into a video...')
        self.logging.emit('Saving the annotated frames into a video...')

        video = cv2.VideoWriter()
        frame_width, frame_height = Image.open(os.path.join(
            '.temp', 'annotated_frame', os.listdir('.temp/annotated_frame')[0])).size

        video.open(os.path.join(self.save_folder, 'annotated_video.mp4'),
                   cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

        self.set_up_progress_bar.emit(len(self.bbox_coordinates))

        for frame in sorted(self.bbox_coordinates.keys(), key=lambda x: int(x.split('.')[0])):
            self.increase_progress_bar.emit()

            frame_path = os.path.join('.temp', 'annotated_frame', frame)
            core_logger.info(f"==>> frame_path: {frame_path}")
            frame_image = cv2.imread(frame_path)
            video.write(frame_image)

        video.release()
        core_logger.info('Video Processing Complete')

        # Reset the progress bar
        self.set_up_progress_bar.emit(0)

        # Clean up the .temp folder
        shutil.rmtree('.temp')

        self.finished.emit()


if __name__ == '__main__':
    worker = ProcessVideoWorker('assets/test_vid.mp4', 'assets')
    worker.run()
