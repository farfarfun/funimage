import base64
import os
from enum import Enum
from io import BytesIO

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
    UNKNOWN = 100000
    CV = 100010
    OSS = 100020
    URL = 100030
    PIL = 100040
    FILE = 100050
    BYTES = 100060
    BASE64 = 100070
    BASE64_STR = 100071
    NDARRAY = 100080
    BYTESIO = 100090


def convert_url_to_bytes(url):
    headers = header.generate()
    try:
        return requests.get(url, headers=headers).content
    except Exception as e:
        logger.error(e)
    try:
        import urllib.request

        return urllib.request.urlopen(url).read()
    except Exception as e:
        logger.error(e)


def parse_image_type(image, image_type=None, *args, **kwargs) -> ImageType:
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


def convert_to_bytes(image, image_type=None, *args, **kwargs):
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
    raise ValueError("Image should be a URL linking to an image, a local path, or a PIL image.")


def convert_to_file(image, image_path, image_type=None, *args, **kwargs):
    return open(image_path, "wb").write(convert_to_bytes(image, image_type, *args, **kwargs))


def convert_to_cvimg(image, image_type=None, *args, **kwargs):
    image_type = parse_image_type(image, image_type, *args, **kwargs)
    if image_type == ImageType.PIL:
        return np.asarray(image)
    if image_type == ImageType.NDARRAY:
        return image
    if image_type == ImageType.CV:
        return image

    try:
        import cv2

        res = cv2.imdecode(np.frombuffer(convert_to_bytes(image), np.uint8), cv2.IMREAD_COLOR)
        assert res is not None
        return res
    except Exception as e:
        logger.error(f"error:{e}")
        PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True
        return np.asarray(PIL.Image.open(BytesIO(convert_to_bytes(image))).convert("RGB"))


def convert_to_pilimg(image, image_type=None, *args, **kwargs):
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
    return PIL.ImageOps.exif_transpose(PIL.Image.open(BytesIO(convert_to_bytes(image)))).convert("RGB")


def convert_to_byte_io(image, image_type=None, *args, **kwargs):
    return BytesIO(convert_to_bytes(image, image_type, *args, **kwargs))


def convert_to_base64(image, image_type=None, *args, **kwargs):
    return base64.b64encode(convert_to_bytes(image, image_type, *args, **kwargs))


def convert_to_base64_str(image, image_type=None, *args, **kwargs):
    image_type = parse_image_type(image, image_type, *args, **kwargs)
    if image_type == ImageType.BASE64_STR:
        return image
    return str(convert_to_base64(image, image_type), encoding="utf-8")
