import cv2
import numpy as np
from matplotlib import pyplot as plt


class DisplayUtility:
    @staticmethod
    def display_features(image, image_key_points):
        output_image = np.array(image)
        for kp in image_key_points:
            cv2.circle(output_image, tuple([int(k) for k in kp]), 10, (0, 0, 255), 3)

        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.imshow("image", output_image)
        cv2.waitKey()

    @staticmethod
    def display_feature_matches(image, image1_key_points, image2_key_points):
        output_image = np.array(image)
        key_points = zip(image1_key_points, image2_key_points)
        for kp in key_points:
            cv2.line(
                output_image,
                tuple([int(k) for k in kp[0]]),
                tuple([int(k) for k in kp[1]]),
                color=(255, 0, 0),
            )
            cv2.circle(output_image, tuple([int(k) for k in kp[0]]), 10, (0, 0, 255), 3)
            cv2.circle(output_image, tuple([int(k) for k in kp[1]]), 10, (0, 0, 155), 3)

        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.imshow("image", output_image)
        cv2.waitKey()

    @staticmethod
    def plot_coordinates(coordinates):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            coordinates[:, 2],
            marker="o",
            edgecolors="none",
            c=coordinates[:, 2],
            cmap="plasma",
        )
        plt.show()
