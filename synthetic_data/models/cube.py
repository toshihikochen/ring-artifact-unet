from typing import Tuple, List, Optional

import numpy as np

from .base import PhantomGenerator


class Cube(PhantomGenerator):
    """
    A cube with bubbles inside.
    """
    def generate(
            self, size: int = 256, height: int = 100, width: int = 100, depth: int = 100,
            bubbles: Optional[List[Tuple[int, int, int, float]]] = None
    ):
        """ Generate a cube with bubbles inside.

        Args:
            size: the size of the voxel.
            height: the height of the cube.
            width: the width of the cube.
            depth: the depth of the cube.
            bubbles: a list of bubbles, each bubble is a tuple of (x, y, z, radius).

        Returns:
            shape=(size, size, size), dtype=uint8, 0 or 1
        """
        if bubbles is None:
            bubbles = [(128, 128, 128, 10), ]

        voxel = self.get_blank_voxel(size)

        z, y, x = np.ogrid[:size, :size, :size]
        center = size // 2

        x_mask = np.abs(x - center) <= width // 2
        y_mask = np.abs(y - center) <= height // 2
        z_mask = np.abs(z - center) <= depth // 2
        cube_mask = x_mask & y_mask & z_mask

        for bubble in bubbles:
            x_bubble, y_bubble, z_bubble, radius = bubble
            dx = x - x_bubble
            dy = y - y_bubble
            dz = z - z_bubble
            bubble_mask = dx ** 2 + dy ** 2 + dz ** 2 <= radius ** 2
            cube_mask &= ~bubble_mask

        voxel[cube_mask] = 1
        return voxel
