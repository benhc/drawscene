from cv2 import undistort


class Rectifier:
    def __init__(self, camera_matrix, distortion_coefficients):
        self._camera_matrix = camera_matrix
        self._distortion_coefficients = distortion_coefficients

    def rectify_image(self, image):
        return undistort(image, self._camera_matrix, self._distortion_coefficients)
