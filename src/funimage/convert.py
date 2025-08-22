"""Image conversion utilities for various formats and types."""

import base64
import os
from enum import Enum
from io import BytesIO
from typing import Any, Optional, Union

import funutil
import numpy as np
import PIL
import PIL.Image
import PIL.ImageFile
import PIL.ImageOps
import pillow_avif
import requests
from funfake.headers import Headers

logger = funutil.getLogger("funimage")

logger.info(f"pillow_avif={pillow_avif.__version__}")
header = Headers()


class ImageType(Enum):
    """Enumeration of supported image input types."""

    UNKNOWN = 100000
    CV = 100010  # OpenCV image
    OSS = 100020  # OSS path (Object Storage Service)
    URL = 100030  # HTTP/HTTPS URL
    PIL = 100040  # PIL Image object
    FILE = 100050  # Local file path
    BYTES = 100060  # Raw bytes
    BASE64 = 100070  # Base64 encoded bytes
    BASE64_STR = 100071  # Base64 encoded string
    NDARRAY = 100080  # NumPy array
    BYTESIO = 100090  # BytesIO object


def convert_url_to_bytes(url: str) -> Optional[bytes]:
    """Convert URL to bytes by downloading the image.

    Args:
        url: HTTP/HTTPS URL to download

    Returns:
        Image bytes or None if download fails
    """
    headers = header.generate()
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.content
    except Exception as e:
        logger.error(f"Failed to download from {url} with requests: {e}")

    try:
        import urllib.request

        return urllib.request.urlopen(url, timeout=30).read()
    except Exception as e:
        logger.error(f"Failed to download from {url} with urllib: {e}")
        return None


def parse_image_type(
    image: Any, image_type: Optional[ImageType] = None, *args, **kwargs
) -> ImageType:
    """Parse and determine the type of input image.

    Args:
        image: Input image in various formats
        image_type: Explicit image type override

    Returns:
        Detected or specified ImageType

    Raises:
        ValueError: If image_type is not an ImageType enum
    """
    if image_type is not None:
        if not isinstance(image_type, ImageType):
            raise ValueError("image_type should be an ImageType Enum.")
        return image_type
    if isinstance(image, PIL.Image.Image):
        return ImageType.PIL
    elif isinstance(image, np.ndarray):
        return ImageType.NDARRAY
    elif isinstance(image, str) and image.startswith("http"):
        return ImageType.URL
    elif isinstance(image, str) and os.path.isfile(image):
        return ImageType.FILE
    elif isinstance(image, str) and image.startswith("{") and "oss_path" in image:
        return ImageType.OSS  # oss
    elif isinstance(image, str):
        return ImageType.BASE64_STR
    elif isinstance(image, bytes):
        return ImageType.BYTES
    elif isinstance(image, BytesIO):
        return ImageType.BYTESIO
    else:
        return ImageType.UNKNOWN


def convert_to_bytes(
    image: Any, image_type: Optional[ImageType] = None, *args, **kwargs
) -> bytes:
    """Convert various image formats to bytes.

    Args:
        image: Input image in various formats
        image_type: Explicit image type override

    Returns:
        Image as bytes

    Raises:
        ValueError: If image format is not supported
    """
    image_type = parse_image_type(image, image_type, *args, **kwargs)
    if image_type == ImageType.URL:
        return convert_url_to_bytes(image)
    if image_type == ImageType.FILE:
        return open(image, "rb").read()
    if image_type == ImageType.BYTES:
        return image
    if image_type == ImageType.BASE64:
        return base64.b64decode(image)
    if image_type == ImageType.PIL:
        image_data = BytesIO()
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        image.save(image_data, format="JPEG")
        return image_data.getvalue()
    if image_type == ImageType.NDARRAY:
        import cv2

        return cv2.imencode(".jpg", image)[1].tobytes()
    if image_type == ImageType.CV:
        import cv2

        return cv2.imencode(".jpg", image)[1]
    raise ValueError(
        f"Unsupported image type: {image_type}. "
        "Image should be a URL, local path, PIL image, bytes, or numpy array."
    )


def convert_to_file(
    image: Any, image_path: str, image_type: Optional[ImageType] = None, *args, **kwargs
) -> int:
    """Convert image to file and save to disk.

    Args:
        image: Input image in various formats
        image_path: Output file path
        image_type: Explicit image type override

    Returns:
        Number of bytes written
    """
    image_bytes = convert_to_bytes(image, image_type, *args, **kwargs)
    with open(image_path, "wb") as f:
        return f.write(image_bytes)


def convert_to_cvimg(
    image: Any, image_type: Optional[ImageType] = None, *args, **kwargs
) -> np.ndarray:
    """Convert image to OpenCV format (numpy array).

    Args:
        image: Input image in various formats
        image_type: Explicit image type override

    Returns:
        Image as numpy array in BGR format
    """
    image_type = parse_image_type(image, image_type, *args, **kwargs)
    if image_type == ImageType.PIL:
        return np.asarray(image)
    if image_type == ImageType.NDARRAY:
        return image
    if image_type == ImageType.CV:
        return image

    try:
        import cv2

        res = cv2.imdecode(
            np.frombuffer(convert_to_bytes(image), np.uint8), cv2.IMREAD_COLOR
        )
        assert res is not None
        return res
    except Exception as e:
        logger.error(f"error:{e}")
        PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True
        return np.asarray(
            PIL.Image.open(BytesIO(convert_to_bytes(image))).convert("RGB")
        )


def convert_to_pilimg(
    image: Any, image_type: Optional[ImageType] = None, *args, **kwargs
) -> PIL.Image.Image:
    """Convert image to PIL Image format.

    Args:
        image: Input image in various formats
        image_type: Explicit image type override

    Returns:
        PIL Image object in RGB format
    """
    image_type = parse_image_type(image, image_type, *args, **kwargs)
    if image_type == ImageType.URL:
        return PIL.Image.open(requests.get(image, stream=True).raw).convert("RGB")
    if image_type == ImageType.FILE:
        return PIL.ImageOps.exif_transpose(PIL.Image.open(image)).convert("RGB")
    if image_type == ImageType.PIL:
        return PIL.ImageOps.exif_transpose(image).convert("RGB")
    if image_type in (ImageType.NDARRAY, ImageType.CV):
        return PIL.ImageOps.exif_transpose(PIL.Image.fromarray(image)).convert("RGB")
    PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True
    return PIL.ImageOps.exif_transpose(
        PIL.Image.open(BytesIO(convert_to_bytes(image)))
    ).convert("RGB")


def convert_to_byte_io(
    image: Any, image_type: Optional[ImageType] = None, *args, **kwargs
) -> BytesIO:
    """Convert image to BytesIO object.

    Args:
        image: Input image in various formats
        image_type: Explicit image type override

    Returns:
        BytesIO object containing image bytes
    """
    return BytesIO(convert_to_bytes(image, image_type, *args, **kwargs))


def convert_to_base64(
    image: Any, image_type: Optional[ImageType] = None, *args, **kwargs
) -> bytes:
    """Convert image to base64 encoded bytes.

    Args:
        image: Input image in various formats
        image_type: Explicit image type override

    Returns:
        Base64 encoded bytes
    """
    return base64.b64encode(convert_to_bytes(image, image_type, *args, **kwargs))


def convert_to_base64_str(
    image: Any, image_type: Optional[ImageType] = None, *args, **kwargs
) -> str:
    """Convert image to base64 encoded string.

    Args:
        image: Input image in various formats
        image_type: Explicit image type override

    Returns:
        Base64 encoded string
    """
    image_type = parse_image_type(image, image_type, *args, **kwargs)
    if image_type == ImageType.BASE64_STR:
        return image
    return convert_to_base64(image, image_type, *args, **kwargs).decode("utf-8")
