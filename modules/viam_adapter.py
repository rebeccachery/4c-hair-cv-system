from io import BytesIO

from PIL import Image

from .pipeline import run_hair_pipeline


def viam_image_to_pil(frame):
    """Convert a ViamImage or NamedImage payload to a RGB PIL image."""
    data = getattr(frame, "data", frame)
    if isinstance(data, str):
        raise TypeError("Expected Viam image bytes, received a string.")
    return Image.open(BytesIO(bytes(data))).convert("RGB")


async def analyze_camera(robot, camera_name, mime_type="image/jpeg"):
    """
    Capture one image from a Viam camera component and run the hair pipeline.

    Viam's current Python SDK exposes Camera.from_robot(...).get_image(),
    returning a ViamImage with byte data. This adapter keeps SDK imports local
    so the CV pipeline remains usable without viam-sdk installed.
    """
    from viam.components.camera import Camera

    camera = Camera.from_robot(robot, camera_name)
    frame = await camera.get_image(mime_type=mime_type)
    try:
        image = viam_image_to_pil(frame)
        return run_hair_pipeline(image)
    finally:
        close = getattr(frame, "close", None)
        if callable(close):
            close()


async def analyze_camera_images(robot, camera_name):
    """
    Capture using get_images(), which is useful for cameras that expose named streams.
    """
    from viam.components.camera import Camera

    camera = Camera.from_robot(robot, camera_name)
    images, metadata = await camera.get_images()
    if not images:
        raise RuntimeError(f"Camera {camera_name!r} returned no images.")

    result = run_hair_pipeline(viam_image_to_pil(images[0]))
    result["viam"] = {
        "camera_name": camera_name,
        "source_name": getattr(images[0], "source_name", None),
        "captured_at": str(getattr(metadata, "captured_at", "")),
    }
    return result
