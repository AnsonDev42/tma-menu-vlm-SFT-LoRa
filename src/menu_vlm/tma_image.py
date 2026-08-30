"""Pinned public reimplementation of the approved current-TMA image processor."""

import importlib
from io import BytesIO
from typing import Any

cv2: Any = importlib.import_module("cv2")
np: Any = importlib.import_module("numpy")
Image: Any = importlib.import_module("PIL.Image")
ImageOps: Any = importlib.import_module("PIL.ImageOps")
register_heif_opener: Any = importlib.import_module("pillow_heif").register_heif_opener

MAX_IMAGE_WIDTH = 2000
MAX_IMAGE_SIZE = 4 * 1024 * 1024 - 100
MAX_IMAGE_PIXELS = 40_000_000

register_heif_opener(thumbnails=False)


def process_tma_image(image: bytes) -> tuple[bytes, int, int]:
    """Produce the JPEG bytes and dimensions used by the approved TMA runtime."""
    decoded = _decode_image_to_bgr(image)
    height, width, _channels = decoded.shape
    if width > MAX_IMAGE_WIDTH:
        scale_ratio = MAX_IMAGE_WIDTH / width
        width = int(width * scale_ratio)
        height = int(height * scale_ratio)
        decoded = cv2.resize(decoded, (width, height), interpolation=cv2.INTER_AREA)

    quality = 80
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, encoded = cv2.imencode(".jpeg", decoded, encode_param)
    if not success:
        raise ValueError("Approved TMA image processor could not encode JPEG")
    while encoded.nbytes > MAX_IMAGE_SIZE:
        quality -= 15
        if quality <= 5:
            break
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, encoded = cv2.imencode(".jpeg", decoded, encode_param)
        if not success:
            raise ValueError("Approved TMA image processor could not encode JPEG")
    return encoded.tobytes(), height, width


def _decode_image_to_bgr(image: bytes) -> Any:
    try:
        with Image.open(BytesIO(image)) as pillow_image:
            width, height = pillow_image.size
    except Exception as exc:
        raise ValueError("Approved TMA image processor could not decode image") from exc
    if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
        raise ValueError("Approved TMA image dimensions exceed the supported limit")

    data = np.frombuffer(image, np.uint8)
    decoded = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if decoded is not None:
        return decoded
    try:
        with Image.open(BytesIO(image)) as pillow_image:
            transposed = ImageOps.exif_transpose(pillow_image)
            rgb_array = np.array(transposed.convert("RGB"))
    except Exception as exc:
        raise ValueError("Approved TMA image processor could not decode image") from exc
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
