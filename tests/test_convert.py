"""Tests for funimage conversion functions."""

import base64
import os
import tempfile
from io import BytesIO
from unittest.mock import Mock, patch

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


class TestParseImageType:
    """Test image type parsing."""

    def test_parse_pil_image(self):
        """Test PIL image detection."""
        img = PIL.Image.new("RGB", (100, 100))
        assert parse_image_type(img) == ImageType.PIL

    def test_parse_numpy_array(self):
        """Test numpy array detection."""
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        assert parse_image_type(arr) == ImageType.NDARRAY

    def test_parse_url(self):
        """Test URL detection."""
        url = "https://example.com/image.jpg"
        assert parse_image_type(url) == ImageType.URL

    def test_parse_bytes(self):
        """Test bytes detection."""
        data = b"fake image data"
        assert parse_image_type(data) == ImageType.BYTES

    def test_parse_bytesio(self):
        """Test BytesIO detection."""
        bio = BytesIO(b"fake image data")
        assert parse_image_type(bio) == ImageType.BYTESIO

    def test_parse_base64_string(self):
        """Test base64 string detection."""
        b64_str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        assert parse_image_type(b64_str) == ImageType.BASE64_STR

    def test_explicit_type_override(self):
        """Test explicit type specification."""
        data = b"fake image data"
        assert parse_image_type(data, ImageType.BASE64) == ImageType.BASE64

    def test_invalid_type_override(self):
        """Test invalid type specification."""
        with pytest.raises(ValueError, match="image_type should be an ImageType Enum"):
            parse_image_type("test", "invalid_type")


class TestConvertToBytes:
    """Test conversion to bytes."""

    def test_convert_pil_to_bytes(self):
        """Test PIL image to bytes conversion."""
        img = PIL.Image.new("RGB", (100, 100), color="red")
        result = convert_to_bytes(img)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_convert_bytes_to_bytes(self):
        """Test bytes passthrough."""
        data = b"fake image data"
        result = convert_to_bytes(data)
        assert result == data

    def test_convert_base64_to_bytes(self):
        """Test base64 to bytes conversion."""
        original = b"test data"
        b64_data = base64.b64encode(original)
        result = convert_to_bytes(b64_data, ImageType.BASE64)
        assert result == original

    @patch("funimage.convert.convert_url_to_bytes")
    def test_convert_url_to_bytes(self, mock_url_convert):
        """Test URL to bytes conversion."""
        mock_url_convert.return_value = b"fake image data"
        url = "https://example.com/image.jpg"
        result = convert_to_bytes(url)
        assert result == b"fake image data"
        mock_url_convert.assert_called_once_with(url)


class TestConvertToPilImg:
    """Test conversion to PIL Image."""

    def test_convert_bytes_to_pil(self):
        """Test bytes to PIL conversion."""
        # Create a simple PNG image in bytes
        img = PIL.Image.new("RGB", (10, 10), color="blue")
        bio = BytesIO()
        img.save(bio, format="PNG")
        img_bytes = bio.getvalue()

        result = convert_to_pilimg(img_bytes)
        assert isinstance(result, PIL.Image.Image)
        assert result.mode == "RGB"
        assert result.size == (10, 10)

    def test_convert_pil_to_pil(self):
        """Test PIL image passthrough with conversion."""
        img = PIL.Image.new("RGBA", (50, 50), color="green")
        result = convert_to_pilimg(img)
        assert isinstance(result, PIL.Image.Image)
        assert result.mode == "RGB"  # Should be converted to RGB


class TestConvertToFile:
    """Test conversion to file."""

    def test_convert_pil_to_file(self):
        """Test PIL image to file conversion."""
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
    """Test base64 conversion functions."""

    def test_convert_to_base64(self):
        """Test conversion to base64 bytes."""
        data = b"test data"
        result = convert_to_base64(data, ImageType.BYTES)
        expected = base64.b64encode(data)
        assert result == expected

    def test_convert_to_base64_str(self):
        """Test conversion to base64 string."""
        data = b"test data"
        result = convert_to_base64_str(data, ImageType.BYTES)
        expected = base64.b64encode(data).decode("utf-8")
        assert result == expected

    def test_base64_str_passthrough(self):
        """Test base64 string passthrough."""
        b64_str = "dGVzdCBkYXRh"  # "test data" in base64
        result = convert_to_base64_str(b64_str)
        assert result == b64_str


class TestConvertToByteIO:
    """Test BytesIO conversion."""

    def test_convert_to_byte_io(self):
        """Test conversion to BytesIO."""
        data = b"test data"
        result = convert_to_byte_io(data, ImageType.BYTES)
        assert isinstance(result, BytesIO)
        assert result.getvalue() == data


class TestUrlToBytes:
    """Test URL download functionality."""

    @patch("requests.get")
    def test_successful_download(self, mock_get):
        """Test successful URL download."""
        mock_response = Mock()
        mock_response.content = b"fake image data"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = convert_url_to_bytes("https://example.com/image.jpg")
        assert result == b"fake image data"

    @patch("requests.get")
    @patch("urllib.request.urlopen")
    def test_fallback_to_urllib(self, mock_urlopen, mock_get):
        """Test fallback to urllib when requests fails."""
        mock_get.side_effect = Exception("Request failed")

        mock_urllib_response = Mock()
        mock_urllib_response.read.return_value = b"urllib image data"
        mock_urlopen.return_value = mock_urllib_response

        result = convert_url_to_bytes("https://example.com/image.jpg")
        assert result == b"urllib image data"

    @patch("requests.get")
    @patch("urllib.request.urlopen")
    def test_both_methods_fail(self, mock_urlopen, mock_get):
        """Test when both download methods fail."""
        mock_get.side_effect = Exception("Request failed")
        mock_urlopen.side_effect = Exception("Urllib failed")

        result = convert_url_to_bytes("https://example.com/image.jpg")
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__])
