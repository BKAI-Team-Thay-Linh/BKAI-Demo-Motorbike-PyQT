import os
import sys
sys.path.append(os.getcwd())  # NOQA

import cv2
import json
import numpy as np

from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet


class DeepSort:
    def __init__(
        self,
        model_path: str,
        max_cosine_distance: float = 0.7,
        nn_budget=None,
        classes=['xeso', 'xega']
    ):
        self.encoder = gdet.create_box_encoder(model_path, batch_size=1)
        self.metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget
        )
        self.tracker = Tracker(self.metric)

        key_list = []  # list of keys
        val_list = []  # list of values

        for idx, class_name in enumerate(classes):
            key_list.append(idx)
            val_list.append(class_name)

        self.key_list = key_list
        self.val_list = val_list

    def tracking(self, origin_frame, bboxes, scores, class_ids):
        features = self.encoder(origin_frame, bboxes)  # Generate features
        detections = [
            Detection(bbox, score, class_id, feature)
            for bbox, score, class_id, feature in zip(bboxes, scores, class_ids, features)
        ]

        self.tracker.predict()
        self.tracker.update(detections)  # Update tracker with current detections

        tracked_bboxes = []
        for track in self.tracker.tracks:
            # If track is confirmed and has been detected for at least 5 frames, draw and label the bounding box
            if not track.is_confirmed() or track.time_since_update > 5:
                continue

            bbox = track.to_tlbr()  # Get the corrected/predicted bounding box
            class_id = track.get_class()
            conf_score = track.get_conf_score()
            tracking_id = track.track_id
            tracked_bboxes.append(
                bbox.tolist() + [class_id, conf_score, tracking_id]
            )

        tracked_bboxes = np.array(tracked_bboxes)

        return tracked_bboxes
