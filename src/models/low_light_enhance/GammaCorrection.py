# import the necessary packages
from __future__ import print_function

import cv2
import numpy as np


def evaluate_image_quality(image):
    brightness = cv2.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))[0]
    return brightness


def find_best_gamma(image, gamma_values):
    best_gamma = None
    best_quality = float("-inf")

    for gamma in gamma_values:
        adjusted_image = adjust_gamma(image, gamma)

        quality = evaluate_image_quality(adjusted_image)

        if quality > best_quality:
            best_quality = quality
            best_gamma = gamma

    return best_gamma


def estimate_gamma(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    hist, bins = np.histogram(gray.flatten(), 256, [0, 256])

    cdf = hist.cumsum()

    cdf_normalized = cdf / cdf.max()

    gamma = np.mean(cdf_normalized)

    return gamma


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)
