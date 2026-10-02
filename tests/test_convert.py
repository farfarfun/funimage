"""测试 funimage 的图像转换函数。"""

import base64
import os
import tempfile
from io import BytesIO
from unittest.mock import patch

import numpy as np
import PIL.Image
import pytest

from funimage import (
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
from funimage.convert import _mask_url


class TestParseImageType:
    """测试图像类型解析。"""

    def test_parse_pil_image(self):
        """测试 PIL 图像检测。"""
        img = PIL.Image.new("RGB", (100, 100))
        assert parse_image_type(img) == ImageType.PIL

    def test_parse_numpy_array(self):
        """测试 NumPy 数组检测。"""
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        assert parse_image_type(arr) == ImageType.NDARRAY

    def test_parse_url(self):
        """测试 URL 检测。"""
        url = "https://example.com/image.jpg"
        assert parse_image_type(url) == ImageType.URL

    def test_parse_bytes(self):
        """测试字节检测。"""
        data = b"fake image data"
        assert parse_image_type(data) == ImageType.BYTES

    def test_parse_bytesio(self):
        """测试 BytesIO 检测。"""
        bio = BytesIO(b"fake image data")
        assert parse_image_type(bio) == ImageType.BYTESIO

    def test_parse_base64_string(self):
        """测试 Base64 字符串检测。"""
        b64_str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        assert parse_image_type(b64_str) == ImageType.BASE64_STR

    def test_explicit_type_override(self):
        """测试显式指定类型。"""
        data = b"fake image data"
        assert parse_image_type(data, ImageType.BASE64) == ImageType.BASE64

    def test_invalid_type_override(self):
        """测试无效的类型指定。"""
        with pytest.raises(ValueError, match="image_type should be an ImageType Enum"):
            parse_image_type("test", "invalid_type")


class TestConvertToBytes:
    """测试转换为字节。"""

    def test_convert_pil_to_bytes(self):
        """测试 PIL 图像转字节。"""
        img = PIL.Image.new("RGB", (100, 100), color="red")
        result = convert_to_bytes(img)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_convert_bytes_to_bytes(self):
        """测试字节原样透传。"""
        data = b"fake image data"
        result = convert_to_bytes(data)
        assert result == data

    def test_convert_base64_to_bytes(self):
        """测试 Base64 转字节。"""
        original = b"test data"
        b64_data = base64.b64encode(original)
        result = convert_to_bytes(b64_data, ImageType.BASE64)
        assert result == original

    def test_convert_base64_string_to_bytes(self):
        """测试 Base64 字符串转字节。"""
        assert convert_to_bytes("dGVzdCBkYXRh") == b"test data"

    def test_convert_bytesio_to_bytes(self):
        """测试 BytesIO 转字节。"""
        assert convert_to_bytes(BytesIO(b"test data")) == b"test data"

    @patch("funimage.convert.convert_url_to_bytes")
    def test_convert_url_to_bytes(self, mock_url_convert):
        """测试 URL 转字节。"""
        mock_url_convert.return_value = b"fake image data"
        url = "https://example.com/image.jpg"
        result = convert_to_bytes(url)
        assert result == b"fake image data"
        mock_url_convert.assert_called_once_with(url)


class TestConvertToPilImg:
    """测试转换为 PIL 图像。"""

    def test_convert_bytes_to_pil(self):
        """测试字节转 PIL 图像。"""
        # 生成一张简单的 PNG 图片字节数据
        img = PIL.Image.new("RGB", (10, 10), color="blue")
        bio = BytesIO()
        img.save(bio, format="PNG")
        img_bytes = bio.getvalue()

        result = convert_to_pilimg(img_bytes)
        assert isinstance(result, PIL.Image.Image)
        assert result.mode == "RGB"
        assert result.size == (10, 10)

    def test_convert_pil_to_pil(self):
        """测试 PIL 图像透传并完成模式转换。"""
        img = PIL.Image.new("RGBA", (50, 50), color="green")
        result = convert_to_pilimg(img)
        assert isinstance(result, PIL.Image.Image)
        assert result.mode == "RGB"  # Should be converted to RGB

    @patch("funimage.convert.convert_url_to_bytes", return_value=None)
    def test_url_download_failure(self, mock_download):
        with pytest.raises(ValueError, match="Failed to download image"):
            convert_to_pilimg("https://example.com/missing.jpg")
        mock_download.assert_called_once()


class TestConvertToCvImg:
    """测试转换为 OpenCV 数组。"""

    def test_convert_pil_image(self):
        """测试 PIL 图像直接转换为 ndarray，无需经过字节编解码。"""
        image = PIL.Image.new("RGB", (2, 3), color="red")
        result = convert_to_cvimg(image)
        assert result.shape == (3, 2, 3)
        assert result[0, 0].tolist() == [255, 0, 0]

    def test_numpy_array_passthrough(self):
        """测试 NumPy 数组原样透传。"""
        image = np.zeros((2, 3, 3), dtype=np.uint8)
        assert convert_to_cvimg(image) is image

    def test_unsupported_input(self):
        """测试不支持的输入类型在字节转换阶段即抛出 ValueError。"""
        with pytest.raises(ValueError, match="Unsupported image type"):
            convert_to_cvimg(object())

    def test_bytes_input_falls_back_to_pillow_without_cv2(self):
        """测试未安装 OpenCV 时，字节输入通过 Pillow 回退成功解码。

        本仓库的测试环境不安装 OpenCV（避免引入重型依赖），因此这里天然覆盖
        `cv2` 导入失败后的 Pillow 回退分支。
        """
        img = PIL.Image.new("RGB", (4, 5), color="blue")
        bio = BytesIO()
        img.save(bio, format="PNG")

        result = convert_to_cvimg(bio.getvalue())

        assert result.shape == (5, 4, 3)

    @patch("funimage.convert.convert_url_to_bytes")
    def test_url_input_downloads_only_once(self, mock_url_convert):
        """测试回退到 Pillow 解码时，URL 只会被下载一次，不会重复触发网络请求。

        这是对 OpenCV 解码失败分支旧实现的回归测试：旧实现在回退前会重新调用
        `convert_to_bytes`，导致 URL 输入被重复下载。
        """
        img = PIL.Image.new("RGB", (3, 3), color="green")
        bio = BytesIO()
        img.save(bio, format="PNG")
        mock_url_convert.return_value = bio.getvalue()

        result = convert_to_cvimg("https://example.com/image.jpg")

        assert result.shape == (3, 3, 3)
        mock_url_convert.assert_called_once()


class TestConvertToFile:
    """测试转换为文件。"""

    def test_convert_pil_to_file(self):
        """测试 PIL 图像转文件。"""
        img = PIL.Image.new("RGB", (20, 20), color="yellow")

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            try:
                bytes_written = convert_to_file(img, tmp.name)
                assert bytes_written > 0
                assert os.path.exists(tmp.name)
                assert os.path.getsize(tmp.name) == bytes_written
            finally:
                os.unlink(tmp.name)


class TestConvertToBase64:
    """测试 Base64 转换相关函数。"""

    def test_convert_to_base64(self):
        """测试转换为 Base64 字节。"""
        data = b"test data"
        result = convert_to_base64(data, ImageType.BYTES)
        expected = base64.b64encode(data)
        assert result == expected

    def test_convert_to_base64_str(self):
        """测试转换为 Base64 字符串。"""
        data = b"test data"
        result = convert_to_base64_str(data, ImageType.BYTES)
        expected = base64.b64encode(data).decode("utf-8")
        assert result == expected

    def test_base64_str_passthrough(self):
        """测试 Base64 字符串原样透传。"""
        b64_str = "dGVzdCBkYXRh"  # "test data" in base64
        result = convert_to_base64_str(b64_str)
        assert result == b64_str


class TestConvertToByteIO:
    """测试转换为 BytesIO。"""

    def test_convert_to_byte_io(self):
        """测试转换为 BytesIO 对象。"""
        data = b"test data"
        result = convert_to_byte_io(data, ImageType.BYTES)
        assert isinstance(result, BytesIO)
        assert result.getvalue() == data


class TestMaskUrl:
    """测试 URL 脱敏辅助函数。"""

    def test_strips_query_and_fragment(self):
        """测试脱敏后丢弃 userinfo、query 与 fragment，仅保留 scheme/host/path。"""
        url = "https://user:pass@example.com/path/image.jpg?token=SECRET#frag"
        assert _mask_url(url) == "https://example.com/path/image.jpg"

    def test_plain_url_unchanged(self):
        """测试不含 query 的普通 URL 脱敏后保持不变。"""
        url = "https://example.com/image.jpg"
        assert _mask_url(url) == url


class TestUrlToBytes:
    """测试 URL 下载功能。"""

    @patch("funimage.convert.simple_download")
    def test_successful_download(self, mock_download):
        """测试 URL 下载成功的场景。"""

        def download(url, filepath, **kwargs):
            with open(filepath, "wb") as file:
                file.write(b"fake image data")
            return True

        mock_download.side_effect = download

        result = convert_url_to_bytes("https://example.com/image.jpg")
        assert result == b"fake image data"
        _, filepath = mock_download.call_args.args
        assert not os.path.exists(filepath)
        assert mock_download.call_args.kwargs["overwrite"] is True
        assert mock_download.call_args.kwargs["timeout"] == 30

    @patch("funimage.convert.simple_download", return_value=False)
    def test_download_failure_log_masks_credentials(self, mock_download):
        """测试下载失败时日志不会包含 URL 中 query 参数携带的凭据信息。"""
        url = "https://example.com/image.jpg?token=SECRET123"

        with patch("funimage.convert.logger") as mock_logger:
            result = convert_url_to_bytes(url)

        assert result is None
        logged = " ".join(str(call) for call in mock_logger.error.call_args_list)
        assert "SECRET123" not in logged
        assert "token" not in logged

    @patch("funimage.convert.simple_download", return_value=False)
    def test_download_failure(self, mock_download):
        result = convert_url_to_bytes("https://example.com/image.jpg")
        assert result is None
        mock_download.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
