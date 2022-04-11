import cv2
import numpy as np
from itertools import product


class PointConstructor:
    def __init__(self, threshold=1):
        self._estimator = cv2.RANSAC
        self._threshold = threshold

    def get_world_image_points(self, image_1_feature_coordinates, image_2_feature_coordinates, camera_matrix):
        (
            projection_matrix_1,
            projection_matrix_2,
            image_1_feature_coordinates,
            image_2_feature_coordinates,
        ) = self._get_projection_matrix(image_1_feature_coordinates, image_2_feature_coordinates, camera_matrix)
        homogenous_world_coordinates = cv2.triangulatePoints(
            projection_matrix_1, projection_matrix_2, image_1_feature_coordinates.T, image_2_feature_coordinates.T
        ).T

        points = cv2.convertPointsFromHomogeneous(homogenous_world_coordinates)
        return [p[0] for p in points]

    def _get_projection_matrix(self, image1_key_points, image2_key_points, camera_matrix):
        essential_matrix, image1_key_points, image2_key_points = self._estimate_essential_matrix(
            image1_key_points, image2_key_points, camera_matrix
        )

        U, s, Vt = np.linalg.svd(essential_matrix)

        W = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

        # There is a fourfold ambiguity in the projection matrix of the camera.
        possible_rotations = [np.dot(U, np.dot(W, Vt)), np.dot(U, np.dot(np.transpose(W), Vt))]
        possible_translations = [U[:, 2], -U[:, 2]]

        canonical_projection_matrix = np.dot(camera_matrix, np.hstack((np.eye(3), np.zeros((3, 1)))))
        possible_projection_matrices = [
            np.dot(camera_matrix, np.hstack((rotation, translation.reshape(-1, 1))))
            for rotation, translation in product(possible_rotations, possible_translations)
        ]

        assert len(possible_projection_matrices) == 4

        # To resolve the ambiguity, for only one projection matrix will all the points lie in front.
        visible_points = []
        for projection_matrix in possible_projection_matrices:
            visible_points.append(
                self._get_points_in_front(
                    image1_key_points, image2_key_points, canonical_projection_matrix, projection_matrix
                )
            )

        if sum(1 for v in visible_points if v > 0) > 1:
            print(f"Warning: more than 1 projection matrix possible, sums are: {visible_points}")

        return (
            canonical_projection_matrix,
            possible_projection_matrices[np.argmax(visible_points)],
            image1_key_points,
            image2_key_points,
        )

    def _estimate_essential_matrix(self, image1_key_points, image2_key_points, camera_matrix):
        essential_matrix, mask = cv2.findEssentialMat(
            image1_key_points,
            image2_key_points,
            camera_matrix,
            self._estimator,
            threshold=self._threshold,
        )

        valid_points = mask.ravel() == 1
        return essential_matrix, image1_key_points[valid_points], image2_key_points[valid_points]

    @staticmethod
    def _get_points_in_front(
        image_1_key_points, image_2_key_points, camera_1_projection_matrix, camera_2_projection_matrix
    ):
        """
        Gets the number of points which lie in front of both the canonical camera and a camera with a given
        projection matrix.
        :param image_1_key_points: The feature coordinates in the first image.
        :param image_2_key_points: The feature coordinates in the first image.
        :param camera_1_projection_matrix: The projection matrix of the first camera.
        :param camera_2_projection_matrix: The projection matrix of the second camera.
        :return: The number of world points which are in front (i.e. positive z coordinate) of both cameras.
        """
        world_coordinates = cv2.triangulatePoints(
            camera_1_projection_matrix, camera_2_projection_matrix, image_1_key_points.T, image_2_key_points.T, None
        )

        camera_1_coordinates = [np.dot(camera_1_projection_matrix, i) for i in world_coordinates.T]
        camera_2_coordinates = [np.dot(camera_2_projection_matrix, i) for i in world_coordinates.T]

        # Negative z coordinate implies it is behind the camera.
        in_front_of_camera_1 = [True if p[2] > 0 else False for p in camera_1_coordinates]
        in_front_of_camera_2 = [True if p[2] > 0 else False for p in camera_2_coordinates]

        return sum([a and b for a, b in zip(in_front_of_camera_1, in_front_of_camera_2)])

    @staticmethod
    def _convert_from_homogeneous(points):
        return points
