from abc import ABC, abstractmethod
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def draw_voxel(voxel: np.ndarray, save_path: Optional[str] = 'voxel.png', downsample_rate: int = 4, show: bool = False):
    """ Draw voxel using matplotlib

    Args:
        voxel: shape=(size, size, size), dtype=uint8, 0 or 1.
        save_path: the path to save the figure.
        downsample_rate: the rate to downsample the voxel, since the voxel is too large to draw.
        show: whether to show the figure.

    Returns:

    """
    voxel = voxel[::downsample_rate, ::downsample_rate, ::downsample_rate]  # downsample

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.voxels(voxel > 0, facecolors=(0.1, 0.5, 0.9, 0.6), edgecolors='black', linewidth=0.3)

    if save_path is not None:
        fig.savefig(save_path)
    if show:
        plt.show()


class PhantomGenerator(ABC):

    @abstractmethod
    def generate(self, size: int = 256, **kwargs):
        """ Generate voxel data.

        Args:
            size: the size of the voxel.
            **kwargs: the parameters of the phantom.

        Returns:
            shape=(size, size, size), dtype=uint8, 0 or 1
        """
        pass

    @staticmethod
    def get_blank_voxel(size: int = 256) -> np.ndarray:
        """ Get a blank voxel.

        Args:
            size: the size of the voxel.

        Returns:
            shape=(size, size, size), dtype=uint8, 0 or 1
        """
        return np.zeros((size, size, size), dtype=np.uint8)

    def show(self, downsample_rate: int = 4, size: int = 256, **kwargs):
        """ Show the phantom.

        Args:
            downsample_rate: the rate to downsample the voxel.
            size: the size of the voxel.
            **kwargs: the parameters of the phantom.
        """
        voxel = self.generate(size, **kwargs)
        draw_voxel(voxel, save_path=None, downsample_rate=downsample_rate, show=True)
