from scenedraw.image_generation import IImageGenerator
from typing import List, Iterator
import numpy as np
import cv2


class FileImageGenerator(IImageGenerator):
    """
    An image generator that yields the contents of supplied image files.
    """

    def __init__(self, file_names: List[str]):
        """
        :param file_names: The ordered list of paths to the image files which should be returned by the generator.
        """
        self._file_names = file_names
        pass

    @property
    def image_stream(self) -> Iterator[np.ndarray]:
        """
        Gets the image stream.
        :return: An image stream derived from the contents of the files.
        """
        for file in self._file_names:
            image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            yield cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
