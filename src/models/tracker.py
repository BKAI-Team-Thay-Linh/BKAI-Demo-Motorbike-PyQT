import os
import sys

sys.path.append(os.getcwd())

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
        classes=["xeso", "xega"],
    ):
        self.encoder = gdet.create_box_encoder(model_path, batch_size=1)
        self.metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget
        )
        self.tracker = Tracker(self.metric)

        # Reduce the max_age to 20
        self.tracker.max_age = 20

        key_list = []  # list of keys
        val_list = []  # list of values

        for idx, class_name in enumerate(classes):
            key_list.append(idx)
            val_list.append(class_name)

        self.key_list = key_list
        self.val_list = val_list

    def tracking(self, origin_frame, bboxes, scores, class_ids):
        features = self.encoder(origin_frame, bboxes)  # Generate features

        # Here, the bounding boxes has already in format (x1, y1, x2, y2)
        # But the deepsort requires the format (x1, y1, w, h)
        # So, we need to convert the bounding boxes to the required format

        bboxes_tlwh = np.array(
            [
                [
                    bbox[0],
                    bbox[1],
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1],
                ]
                for bbox in bboxes
            ]
        )

        detections = [
            Detection(bbox, score, class_id, feature)
            for bbox, score, class_id, feature in zip(
                bboxes_tlwh, scores, class_ids, features
            )
        ]

        self.tracker.predict()
        self.tracker.update(detections)  # Update tracker with current detections

        tracked_bboxes = []
        for track in self.tracker.tracks:
            # If track is not confirmed or it's has been 8 frames since its last update, skip
            if not track.is_confirmed() or track.time_since_update > 8:
                continue

            bbox = track.to_tlbr()  # Get the corrected/predicted bounding box
            class_id = track.get_class()
            conf_score = track.get_conf_score()
            tracking_id = track.track_id
            tracked_bboxes.append(bbox.tolist() + [class_id, conf_score, tracking_id])

        tracked_bboxes = np.array(tracked_bboxes)

        return tracked_bboxes
