"""FunImage 提供 PIL、OpenCV、字节、Base64、URL 和文件之间的图像转换。"""

from importlib.metadata import version as _version

from .convert import (
    ImageType,
    convert_to_base64,
    convert_to_base64_str,
    convert_to_byte_io,
    convert_to_bytes,
    convert_to_cvimg,
    convert_to_file,
    convert_to_pilimg,
    convert_url_to_bytes,
    parse_image_type,
)

__version__ = _version("funimage")
__author__ = "farfarfun"
__email__ = "farfarfun@qq.com"

__all__ = [
    "ImageType",
    "convert_to_base64",
    "convert_to_base64_str",
    "convert_to_byte_io",
    "convert_to_bytes",
    "convert_to_cvimg",
    "convert_to_file",
    "convert_to_pilimg",
    "convert_url_to_bytes",
    "parse_image_type",
]
