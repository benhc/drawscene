from cv2 import FastFeatureDetector_create, calcOpticalFlowPyrLK
import numpy as np


class FastOpticalFlowFeatureMatcher:
    """
    A feature matcher which uses FAST feature detection and pyramidal Lukas-Kanade optical flow to generate matched
    features.
    """

    def __init__(self, error_threshold):
        self._error_threshold = error_threshold
        self._fast_feature_detector = FastFeatureDetector_create()

    def get_matched_features(self, image1, image2):
        image1_key_points, image2_key_points, error = self._create_features(image1, image2)
        return self._threshold_error(image1_key_points, image2_key_points, error)

    def _create_features(self, image1, image2):
        image1_features = self._fast_feature_detector.detect(image1, None)
        image1_key_points = np.float32(np.array([p.pt for p in image1_features]))

        image2_key_points, status, err = calcOpticalFlowPyrLK(image1, image2, image1_key_points, None)
        status_mask = [s == 1 for s in status.ravel()]
        return image1_key_points[status_mask], image2_key_points[status_mask], err[status_mask]

    def _threshold_error(self, image1_key_points, image2_key_points, error):
        below_error_threshold = np.ravel(error < self._error_threshold)
        return (
            image1_key_points[below_error_threshold],
            image2_key_points[below_error_threshold],
        )
