"""
Ground truth liquid glass renderer using Snell's law refraction.

Generates displacement and specular maps for a sphere-cap glass surface
with optional sine-wave perturbations for a wobbly look.
"""

import numpy as np
import cv2
from pathlib import Path


def make_sphere_cap_surface(
    size: int = 256,
    radius_frac: float = 0.4,
    height: float = 0.15,
    wobble_freq: float = 0.0,
    wobble_amp: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a height map for a sphere-cap surface centered in the frame.

    Args:
        size: Image dimension (square).
        radius_frac: Radius of the glass circle as fraction of image size.
        height: Peak height of the dome (controls curvature / refraction strength).
        wobble_freq: Frequency of sine perturbations (0 = smooth sphere).
        wobble_amp: Amplitude of sine perturbations relative to height.
        seed: Random seed for phase offsets.

    Returns:
        height_map: float32 array [H, W], values in [0, height].
    """
    rng = np.random.RandomState(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    cx, cy = size / 2.0, size / 2.0
    radius = size * radius_frac

    # Normalized distance from center [0, 1] within the circle
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    mask = dist < radius
    dist_norm = np.clip(dist / radius, 0, 1)

    # Sphere cap: h(r) = height * sqrt(1 - r^2)
    h = np.zeros((size, size), dtype=np.float32)
    h[mask] = height * np.sqrt(np.clip(1.0 - dist_norm[mask] ** 2, 0, 1))

    # Optional wobble perturbation
    if wobble_freq > 0 and wobble_amp > 0:
        phase_x = rng.uniform(0, 2 * np.pi)
        phase_y = rng.uniform(0, 2 * np.pi)
        wobble = wobble_amp * height * (
            np.sin(wobble_freq * (x - cx) / radius * 2 * np.pi + phase_x)
            * np.sin(wobble_freq * (y - cy) / radius * 2 * np.pi + phase_y)
        )
        h[mask] += wobble[mask]
        h = np.clip(h, 0, None)

    # Smooth the edge to avoid sharp discontinuity
    edge_width = 0.05
    edge_mask = (dist_norm > 1.0 - edge_width) & mask
    falloff = np.clip((1.0 - dist_norm[edge_mask]) / edge_width, 0, 1)
    h[edge_mask] *= falloff

    return h


def compute_normals(height_map: np.ndarray) -> np.ndarray:
    """
    Compute surface normals from a height map using finite differences.

    Returns:
        normals: float32 array [H, W, 3], unit normals.
    """
    # Gradient in x and y
    dhdx = np.zeros_like(height_map)
    dhdy = np.zeros_like(height_map)
    dhdx[:, 1:-1] = (height_map[:, 2:] - height_map[:, :-2]) / 2.0
    dhdy[1:-1, :] = (height_map[2:, :] - height_map[:-2, :]) / 2.0

    # Normal = (-dh/dx, -dh/dy, 1), normalized
    normals = np.stack([-dhdx, -dhdy, np.ones_like(height_map)], axis=-1)
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals /= np.clip(norms, 1e-8, None)

    return normals


def snell_displacement(
    normals: np.ndarray,
    ior: float = 1.5,
    thickness: float = 1.0,
) -> np.ndarray:
    """
    Compute 2D displacement field using Snell's law refraction.

    For each pixel, the incident ray is straight down (0, 0, -1).
    The refracted ray through the glass surface displaces the sampling point.

    Args:
        normals: Surface normals [H, W, 3].
        ior: Index of refraction (glass ~1.5).
        thickness: Effective thickness multiplier for displacement magnitude.

    Returns:
        displacement: float32 array [H, W, 2] (dx, dy) in pixel coordinates.
    """
    h, w = normals.shape[:2]

    # Incident ray: straight down in screen space = (0, 0, -1)
    incident = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    # Snell's law in vector form
    # n1 * sin(theta_i) = n2 * sin(theta_r)
    n1, n2 = 1.0, ior
    ratio = n1 / n2

    cos_i = -np.sum(normals * incident, axis=-1)  # dot(normal, -incident)
    cos_i = np.clip(cos_i, 0, 1)

    # Check for total internal reflection
    sin2_t = ratio ** 2 * (1.0 - cos_i ** 2)
    valid = sin2_t < 1.0

    cos_t = np.sqrt(np.clip(1.0 - sin2_t, 0, 1))

    # Refracted direction
    refracted = np.zeros_like(normals)
    for c in range(3):
        refracted[:, :, c] = ratio * incident[c] + (ratio * cos_i - cos_t) * normals[:, :, c]

    # Displacement = refracted ray's xy offset * thickness
    # Normalize by z component to get the offset at the "exit" plane
    rz = np.clip(np.abs(refracted[:, :, 2]), 1e-8, None)
    dx = refracted[:, :, 0] / rz * thickness * w * 0.1
    dy = refracted[:, :, 1] / rz * thickness * h * 0.1

    # Zero out invalid (total internal reflection) and outside glass
    dx[~valid] = 0
    dy[~valid] = 0

    displacement = np.stack([dx, dy], axis=-1)
    return displacement


def compute_specular(
    normals: np.ndarray,
    light_dir: np.ndarray = None,
    shininess: float = 64.0,
) -> np.ndarray:
    """
    Compute specular highlights using Blinn-Phong model.

    Args:
        normals: Surface normals [H, W, 3].
        light_dir: Light direction (normalized). Default: upper-left.
        shininess: Specular exponent.

    Returns:
        specular: float32 array [H, W], values in [0, 1].
    """
    if light_dir is None:
        light_dir = np.array([0.3, 0.3, 1.0], dtype=np.float32)
    light_dir = light_dir / np.linalg.norm(light_dir)

    view_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    half_vec = (light_dir + view_dir)
    half_vec = half_vec / np.linalg.norm(half_vec)

    ndoth = np.sum(normals * half_vec, axis=-1)
    ndoth = np.clip(ndoth, 0, 1)
    specular = ndoth ** shininess

    return specular.astype(np.float32)


def apply_displacement(
    image: np.ndarray,
    displacement: np.ndarray,
    specular: np.ndarray,
    specular_intensity: float = 0.5,
) -> np.ndarray:
    """
    Apply displacement field and specular to an image.

    Args:
        image: Input image [H, W, 3], uint8.
        displacement: Displacement field [H, W, 2] in pixels.
        specular: Specular map [H, W], values in [0, 1].
        specular_intensity: Strength of specular highlights.

    Returns:
        composited: Output image [H, W, 3], uint8.
    """
    h, w = image.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)

    map_x = x + displacement[:, :, 0]
    map_y = y + displacement[:, :, 1]

    # Clamp to image bounds
    map_x = np.clip(map_x, 0, w - 1)
    map_y = np.clip(map_y, 0, h - 1)

    # Remap with bilinear interpolation
    result = cv2.remap(
        image, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    # Add specular highlights
    spec_rgb = (specular[:, :, np.newaxis] * specular_intensity * 255).astype(np.float32)
    result = np.clip(result.astype(np.float32) + spec_rgb, 0, 255).astype(np.uint8)

    return result


def generate_ground_truth(
    image: np.ndarray,
    surface_params: dict = None,
) -> dict:
    """
    Full pipeline: image → ground truth displacement, specular, composited.

    Args:
        image: Input image [H, W, 3], uint8, should be square.
        surface_params: Override surface parameters.

    Returns:
        dict with keys: displacement, specular, composited, height_map, normals
    """
    size = image.shape[0]
    params = {
        "size": size,
        "radius_frac": 0.4,
        "height": 0.15,
        "wobble_freq": 0.0,
        "wobble_amp": 0.0,
        "ior": 1.5,
        "thickness": 1.0,
        "shininess": 64.0,
        "specular_intensity": 0.5,
    }
    if surface_params:
        params.update(surface_params)

    height_map = make_sphere_cap_surface(
        size=params["size"],
        radius_frac=params["radius_frac"],
        height=params["height"],
        wobble_freq=params["wobble_freq"],
        wobble_amp=params["wobble_amp"],
    )
    normals = compute_normals(height_map)
    displacement = snell_displacement(
        normals, ior=params["ior"], thickness=params["thickness"]
    )
    specular = compute_specular(normals, shininess=params["shininess"])
    composited = apply_displacement(
        image, displacement, specular,
        specular_intensity=params["specular_intensity"],
    )

    return {
        "displacement": displacement,
        "specular": specular,
        "composited": composited,
        "height_map": height_map,
        "normals": normals,
    }
