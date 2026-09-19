"""不同图像格式和输入类型之间的转换工具。"""

import base64
import os
import tempfile
from enum import Enum
from io import BytesIO
from typing import Any

import numpy as np
import PIL
import PIL.Image
import PIL.ImageFile
import PIL.ImageOps
import pillow_avif  # noqa: F401 - 注册 AVIF 插件
from farlog import getLogger
from funfake.headers import Headers
from funget import simple_download

logger = getLogger("funimage")

header = Headers()


class ImageType(Enum):
    """图像转换支持的输入类型。"""

    UNKNOWN = 100000
    CV = 100010  # OpenCV 图像
    OSS = 100020  # 对象存储路径
    URL = 100030  # HTTP/HTTPS 地址
    PIL = 100040  # PIL 图像对象
    FILE = 100050  # 本地文件路径
    BYTES = 100060  # 原始字节
    BASE64 = 100070  # Base64 编码字节
    BASE64_STR = 100071  # Base64 编码字符串
    NDARRAY = 100080  # NumPy 数组
    BYTESIO = 100090  # BytesIO 对象


def convert_url_to_bytes(url: str) -> bytes | None:
    """下载 URL 指向的图像并返回字节。

    Args:
        url: 待下载的 HTTP/HTTPS 地址。

    Returns:
        图像字节；下载失败时返回 `None`。
    """
    try:
        with tempfile.TemporaryDirectory() as directory:
            filepath = os.path.join(directory, "image")
            if simple_download(
                url,
                filepath,
                overwrite=True,
                headers=header.generate(),
                timeout=30,
            ):
                with open(filepath, "rb") as file:
                    return file.read()
    except OSError as exc:
        logger.error(f"Failed to read downloaded image from {url}: {exc}")
        return None
    logger.error(f"Failed to download image from {url}")
    return None


def parse_image_type(
    image: Any,
    image_type: ImageType | None = None,
    *args: Any,
    **kwargs: Any,
) -> ImageType:
    """识别输入图像的类型。

    Args:
        image: 待识别的图像数据。
        image_type: 显式指定的类型，优先于自动识别。
        *args: 为兼容旧接口保留。
        **kwargs: 为兼容旧接口保留。

    Returns:
        识别或指定的 `ImageType`。

    Raises:
        ValueError: `image_type` 不是 `ImageType` 枚举。
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
        return ImageType.OSS
    elif isinstance(image, str):
        return ImageType.BASE64_STR
    elif isinstance(image, bytes):
        return ImageType.BYTES
    elif isinstance(image, BytesIO):
        return ImageType.BYTESIO
    else:
        return ImageType.UNKNOWN


def convert_to_bytes(
    image: Any,
    image_type: ImageType | None = None,
    *args: Any,
    **kwargs: Any,
) -> bytes:
    """将支持的图像输入转换为字节。

    Args:
        image: 待转换的图像数据。
        image_type: 显式指定的输入类型。
        *args: 传给类型识别的兼容参数。
        **kwargs: 传给类型识别的兼容参数。

    Returns:
        图像字节。

    Raises:
        ValueError: 图像类型不受支持或 URL 下载失败。
    """
    image_type = parse_image_type(image, image_type, *args, **kwargs)
    if image_type == ImageType.URL:
        image_bytes = convert_url_to_bytes(image)
        if image_bytes is None:
            raise ValueError(f"Failed to download image: {image}")
        return image_bytes
    if image_type == ImageType.FILE:
        with open(image, "rb") as file:
            return file.read()
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
    image: Any,
    image_path: str,
    image_type: ImageType | None = None,
    *args: Any,
    **kwargs: Any,
) -> int:
    """将图像转换为字节并写入文件。

    Args:
        image: 待转换的图像数据。
        image_path: 输出文件路径。
        image_type: 显式指定的输入类型。
        *args: 传给字节转换的兼容参数。
        **kwargs: 传给字节转换的兼容参数。

    Returns:
        写入的字节数。
    """
    image_bytes = convert_to_bytes(image, image_type, *args, **kwargs)
    with open(image_path, "wb") as f:
        return f.write(image_bytes)


def convert_to_cvimg(
    image: Any,
    image_type: ImageType | None = None,
    *args: Any,
    **kwargs: Any,
) -> np.ndarray:
    """将图像转换为 OpenCV 可用的 NumPy 数组。

    Args:
        image: 待转换的图像数据。
        image_type: 显式指定的输入类型。
        *args: 传给类型识别的兼容参数。
        **kwargs: 传给类型识别的兼容参数。

    Returns:
        图像 NumPy 数组。
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
    except Exception as exc:  # noqa: BLE001 - OpenCV 解码失败时回退到 Pillow
        logger.error(f"error:{exc}")
        PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True
        return np.asarray(
            PIL.Image.open(BytesIO(convert_to_bytes(image))).convert("RGB")
        )


def convert_to_pilimg(
    image: Any,
    image_type: ImageType | None = None,
    *args: Any,
    **kwargs: Any,
) -> PIL.Image.Image:
    """将图像转换为 RGB 模式的 PIL 图像。

    Args:
        image: 待转换的图像数据。
        image_type: 显式指定的输入类型。
        *args: 传给类型识别的兼容参数。
        **kwargs: 传给类型识别的兼容参数。

    Returns:
        RGB 模式的 PIL 图像对象。
    """
    image_type = parse_image_type(image, image_type, *args, **kwargs)
    if image_type == ImageType.URL:
        image_bytes = convert_url_to_bytes(image)
        if image_bytes is None:
            raise ValueError(f"Failed to download image: {image}")
        return PIL.Image.open(BytesIO(image_bytes)).convert("RGB")
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
    image: Any,
    image_type: ImageType | None = None,
    *args: Any,
    **kwargs: Any,
) -> BytesIO:
    """将图像转换为 `BytesIO` 对象。

    Args:
        image: 待转换的图像数据。
        image_type: 显式指定的输入类型。
        *args: 传给字节转换的兼容参数。
        **kwargs: 传给字节转换的兼容参数。

    Returns:
        包含图像字节的 `BytesIO` 对象。
    """
    return BytesIO(convert_to_bytes(image, image_type, *args, **kwargs))


def convert_to_base64(
    image: Any,
    image_type: ImageType | None = None,
    *args: Any,
    **kwargs: Any,
) -> bytes:
    """将图像转换为 Base64 编码字节。

    Args:
        image: 待转换的图像数据。
        image_type: 显式指定的输入类型。
        *args: 传给字节转换的兼容参数。
        **kwargs: 传给字节转换的兼容参数。

    Returns:
        Base64 编码字节。
    """
    return base64.b64encode(convert_to_bytes(image, image_type, *args, **kwargs))


def convert_to_base64_str(
    image: Any,
    image_type: ImageType | None = None,
    *args: Any,
    **kwargs: Any,
) -> str:
    """将图像转换为 Base64 编码字符串。

    Args:
        image: 待转换的图像数据。
        image_type: 显式指定的输入类型。
        *args: 传给类型识别和字节转换的兼容参数。
        **kwargs: 传给类型识别和字节转换的兼容参数。

    Returns:
        Base64 编码字符串。
    """
    image_type = parse_image_type(image, image_type, *args, **kwargs)
    if image_type == ImageType.BASE64_STR:
        return image
    return convert_to_base64(image, image_type, *args, **kwargs).decode("utf-8")
