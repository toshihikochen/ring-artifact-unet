import importlib
import os

import astra
import numpy as np
import tifffile
from tqdm import tqdm

from models.base import draw_voxel


OUTPUT_DIR = "output"
DEBUG_DIR = "debug"  # set `None` to disable voxel debug output
NUM_ANGLES = 720
NUM_NOISE_VARIANCE = 5

MODELS_TO_GENERATE = [
    {"type": "cube", "params": {"size": 256, "height": 100, "width": 100, "depth": 100, "bubbles": [(128, 128, 128, 10), ]}},
    {"type": "cube", "params": {"size": 256, "height": 200, "width": 180, "depth": 160, "bubbles": [(150, 135, 140, 16), (200, 92, 130, 24), (70, 50, 60, 30), (200, 45, 50, 26), (75, 60, 195, 20), (120, 200, 160, 30)]}},
    {"type": "mobius_strip", "params": {"size": 256, "radius": 100, "width": 20, "thickness": 1.0}},
    {"type": "octahedron", "params": {"size": 256, "scale": 0.7}},
    {"type": "plate", "params": {"size": 256, "thickness": 100, "a": 100, "b": 60, "hole_radius": 15}},
    {"type": "tetrahedron", "params": {"size": 256, "height": 100}},
    {"type": "torus", "params": {"size": 256, "major_r": 0.28, "minor_r": 0.10, "thickness": None}},
    {"type": "vase", "params": {"size": 256, "base_radius": 0.22, "neck_radius": 0.09, "rim_radius": 0.26, "neck_height": 0.38, "total_height": 0.92, "wall_thickness": 0.045}}
]


def get_generator_class(model_type: str):
    try:
        module = importlib.import_module(f"models.{model_type}")
        class_name = "".join([n.capitalize() for n in model_type.split("_")])
        generator_class = getattr(module, class_name)
        return generator_class
    except Exception as e:
        raise e


def project(voxel: np.ndarray) -> np.ndarray:
    size = voxel.shape[0]
    vol_geom = astra.create_vol_geom(size, size, size)
    angles = np.linspace(0, 2 * np.pi, NUM_ANGLES, False)
    proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, size, size, angles)

    vol_id = astra.data3d.create('-vol', vol_geom, voxel)
    proj_id = astra.data3d.create('-proj3d', proj_geom)

    cfg = astra.astra_dict('FP3D_CUDA')
    cfg['VolumeDataId'] = vol_id
    cfg['ProjectionDataId'] = proj_id

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    clean_projections = astra.data3d.get(proj_id)

    astra.algorithm.delete(alg_id)
    astra.data3d.delete(vol_id)
    astra.data3d.delete(proj_id)

    minv, maxv = clean_projections.min(), clean_projections.max()
    if maxv <= minv:
        return np.full_like(clean_projections, 0.11, dtype=np.float32)
    return (0.02 + (clean_projections - minv) / (maxv - minv) * 0.18).astype(np.float32)

def add_noise(sinogram: np.ndarray) -> np.ndarray:
    noisy = sinogram.copy()

    # stripe noise

    def unresponsive(c):
        """-15%~+15%"""
        reduction = np.random.uniform(0.85, 1.15)
        return c * reduction
    
    def full(c):
        """+10%~+20%"""
        intensity = np.random.uniform(0.10, 0.20)
        return c * (1 + intensity)
    
    def partial(c):
        """ """
        start, end = np.random.randint(0, len(c)), np.random.randint(0, len(c))
        intensity = np.random.uniform(0.05, 0.15)
        c[start: end] *= (1 + intensity)
        return c
    
    def fluctuating(c):
        """ """
        amplitude = np.random.uniform(0.05, 0.15)
        noise = np.random.normal(1, amplitude, len(c))
        return c * noise
    
    def noops(c):
        """no ops"""
        return c
    
    functions = (unresponsive, full, partial, fluctuating, noops)
    p = (0.05, 0.2, 0.35, 0.38, 0.02)

    for col in range(noisy.shape[1]):
        f = functions[np.random.choice(5, p=p)]
        noisy[:, col] = f(noisy[:, col])

    # global noise
    amplitude = np.random.uniform(0.005, 0.010)
    noise = np.random.normal(1, amplitude, size=noisy.shape)
    noisy *= noise

    return noisy


def main():
    os.makedirs(os.path.join(OUTPUT_DIR, "label"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "noisy"), exist_ok=True)

    pbar = tqdm(MODELS_TO_GENERATE)
    for i, model_cfg in enumerate(pbar, start=1):
        model_type = model_cfg["type"]
        params = model_cfg.get("params", {})
        run_id = f"{str(i).zfill(3)}-{model_type}"
        pbar.set_description(f"Generating {run_id}")

        generator_class = get_generator_class(model_type)
        generator = generator_class()
        voxel = generator.generate(**params)

        # debug
        # output with a voxel image
        if DEBUG_DIR:
            pbar.set_description(f"Saving {run_id} debug image")
            draw_voxel(voxel, f"{DEBUG_DIR}/{run_id}.png")

        # astra projection
        pbar.set_description(f"Generating {run_id} astra projection")
        projections = project(voxel)

        # save label and noise data in tiff format
        pbar.set_description(f"Saving {run_id} in TIFF")
        for j, proj in enumerate(projections, start=1):
            # if there is barely any structures of the projections, do skip
            if np.std(proj) < 1e-4:
                continue
            # label
            filename = f"{run_id}-{j:04d}.tiff"
            tifffile.imwrite(os.path.join(OUTPUT_DIR, "label", filename), proj)
            # noisy data
            for var_i in range(NUM_NOISE_VARIANCE):
                noisy = add_noise(proj)
                filename = f"{run_id}-{j:04d}-{var_i:03d}.tiff"
                tifffile.imwrite(os.path.join(OUTPUT_DIR, "noisy", filename), noisy)


if __name__ == "__main__":
    main()
