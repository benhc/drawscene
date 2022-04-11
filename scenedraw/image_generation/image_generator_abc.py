from abc import ABC
import numpy as np
from typing import Iterator


class ImageGeneratorABC(ABC):
    """
    Defines the interface for a class which provides a stream of images.
    """

    @property
    def image_stream(self) -> Iterator[np.ndarray]:
        """
        Gets the image stream.
        :return: An iterator of images in the form of an array of pixel values.
        """
        pass
