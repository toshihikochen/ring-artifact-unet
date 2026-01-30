import numpy as np

from .base import PhantomGenerator


class Plate(PhantomGenerator):
    """
    A plate with an elliptical shape and a circular hole in the center.
    """
    def generate(
            self, size: int = 256, thickness: int = 100, a: int = 100, b: int = 60, hole_radius: int = 15
    ):
        """ Generate a plate with an elliptical shape and a circular hole in the center.

        Args:
            size: the size of the voxel.
            thickness: the thickness of the plate.
            a: semi-axis length of the ellipse along the x-axis.
            b: semi-axis length of the ellipse along the y-axis.
            hole_radius: the radius of the circular hole.

        Returns:
            shape=(size, size, size), dtype=uint8, 0 or 1
        """
        voxel = self.get_blank_voxel(size)

        z, y, x = np.ogrid[:size, :size, :size]  # shape: (size, 1, 1) (1, size, 1) (1, 1, size)
        center = size // 2
        dx = x - center
        dy = y - center
        dz = z - center

        z_mask = np.abs(dz) <= thickness // 2
        ellipse_mask = (dx / a) ** 2 + (dy / b) ** 2 <= 1
        hole_mask = dx ** 2 + dy ** 2 <= hole_radius ** 2
        plate_mask = z_mask & ellipse_mask & ~hole_mask

        voxel[plate_mask] = 1
        return voxel
