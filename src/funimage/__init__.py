"""FunImage - A powerful Python library for image format conversion and processing."""

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

__version__ = "1.0.13"
__author__ = "farfarfun"
__email__ = "farfarfun@qq.com"

__all__ = [
    # Core conversion functions
    "convert_to_base64",
    "convert_to_base64_str",
    "convert_to_byte_io",
    "convert_to_bytes",
    "convert_to_cvimg",
    "convert_to_file",
    "convert_to_pilimg",
    "convert_url_to_bytes",
    # Utility functions
    "parse_image_type",
    # Enums
    "ImageType",
]
