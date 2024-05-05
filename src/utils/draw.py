import os
import sys

import cv2
import numpy as np

sys.path.append(os.getcwd())

import src.utils.constants as const


def draw_bboxes(
    img: np.ndarray,
    bboxes: list,  # Bounding boxes
    scores: list,  # Detection scores
    class_ids: list,  # Classify ids
    track_ids: list,  # Tracking ids
) -> np.ndarray:
    np.random.seed(0)
    height, width = img.shape[:2]

    # Create mask image
    mask_img = img.copy()
    det_img = img.copy()

    size = min([height, width]) * 0.0006
    text_thickness = int(min([height, width]) * 0.001)

    # Draw bounding boxes and labels of detections
    for bbox, _, class_id, track_id in zip(bboxes, scores, class_ids, track_ids):
        if class_id <= 2:
            color = const.COLOR[class_id]
            x1, y1, x2, y2 = bbox.astype(int)

            # Draw rectangle on the image and the mask
            cv2.rectangle(
                img=det_img, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=2
            )

            cv2.rectangle(
                img=mask_img, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=-1
            )

            # Get the label and caption
            label = const.MOTOR_CLASSES[class_id]
            caption = f"{label} ID: {track_id}"

            # Get the size of the text
            (tw, th), _ = cv2.getTextSize(
                text=caption,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=size,
                thickness=text_thickness,
            )
            th = int(th * 1.2)  # Add some padding

            cv2.rectangle(
                img=det_img,
                pt1=(x1, y1),
                pt2=(x1 + tw, y1 - th),
                color=color,
                thickness=-1,
            )
            cv2.rectangle(
                img=mask_img,
                pt1=(x1, y1),
                pt2=(x1 + tw, y1 - th),
                color=color,
                thickness=-1,
            )

            for img in [det_img, mask_img]:
                cv2.putText(
                    img,
                    caption,
                    (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    size,
                    (255, 255, 255),
                    text_thickness,
                    cv2.LINE_AA,
                )
        else:
            print(f"Unknown class_id: {class_id}")
            continue

    return cv2.addWeighted(mask_img, const.MASK_ALPHA, det_img, 1 - const.MASK_ALPHA, 0)
