import numpy as np

from scenedraw import FastOpticalFlowFeatureMatcher, Rectifier, PointConstructor
from scenedraw.image_generation import FileImageGenerator
from scenedraw.display import DisplayUtility


class Program:
    """
    The main application.
    """

    @staticmethod
    def run():
        """
        Runs the program.
        """

        image_files = ["0005.jpg", "0004.jpg"]

        camera_matrix = np.array([[2759.48, 0, 1520.69, 0, 2764.16, 1006.81, 0, 0, 1]]).reshape(3, 3)
        distortion_coefficients = np.zeros((1, 5))

        rectifier = Rectifier(camera_matrix, distortion_coefficients)
        image_generator = FileImageGenerator(image_files)
        feature_matcher = FastOpticalFlowFeatureMatcher(error_threshold=10.0)
        point_constructor = PointConstructor(threshold=0.5)

        image_1 = None
        for image in image_generator.image_stream:
            rectified_image = rectifier.rectify_image(image)
            image_2 = rectified_image

            if image_1 is None:
                image_1 = image_2
                continue

            kp1, kp2 = feature_matcher.get_matched_features(image_1, image_2)

            # DisplayUtility.display_feature_matches(image_1, kp1, kp2)

            world_image_points = point_constructor.get_world_image_points(kp1, kp2, camera_matrix)

            print(world_image_points)

            DisplayUtility.plot_coordinates(world_image_points.T)
            image_1 = image_2
