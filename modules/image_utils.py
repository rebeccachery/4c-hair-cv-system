from io import BytesIO

import numpy as np
from PIL import Image


def to_pil_image(image):
    """Convert supported image inputs to a RGB PIL image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, (bytes, bytearray, memoryview)):
        return Image.open(BytesIO(bytes(image))).convert("RGB")

    data = getattr(image, "data", None)
    if data is not None:
        return Image.open(BytesIO(data)).convert("RGB")

    array = np.asarray(image)
    if array.ndim == 2:
        return Image.fromarray(array.astype(np.uint8), mode="L").convert("RGB")
    if array.ndim == 3 and array.shape[2] == 4:
        array = array[:, :, :3].astype(np.uint8)
    elif array.ndim == 3 and array.shape[2] == 3:
        array = array.astype(np.uint8)
    else:
        raise ValueError(f"Unsupported image shape: {array.shape}")

    return Image.fromarray(array).convert("RGB")


def to_rgb_array(image):
    """Convert supported image inputs to a HxWx3 uint8 RGB array."""
    return np.asarray(to_pil_image(image), dtype=np.uint8)


def normalize_mask(mask, image_shape):
    """Return a binary uint8 mask resized to the image height and width."""
    height, width = image_shape[:2]
    if mask is None:
        return np.zeros((height, width), dtype=np.uint8)

    mask_array = np.asarray(mask)
    if mask_array.ndim == 3:
        mask_array = mask_array[:, :, 0]

    mask_image = Image.fromarray(mask_array.astype(np.uint8))
    resized = np.asarray(mask_image.resize((width, height), Image.Resampling.NEAREST))
    return (resized > 0).astype(np.uint8)
