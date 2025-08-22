# FunImage

[![PyPI version](https://badge.fury.io/py/funimage.svg)](https://badge.fury.io/py/funimage)
[![Python Support](https://img.shields.io/pypi/pyversions/funimage.svg)](https://pypi.org/project/funimage/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


一个强大的 Python 图像格式转换和处理库。FunImage 提供了 PIL 图像、OpenCV 数组、字节、base64、URL 和文件路径之间的无缝转换功能。

A powerful Python library for image format conversion and processing. FunImage provides seamless conversion between various image formats including PIL Images, OpenCV arrays, bytes, base64, URLs, and file paths.

## 特性 Features

- 🔄 **通用图像转换 Universal Image Conversion**: 支持 PIL、OpenCV、字节、base64、URL 和文件之间的转换 Convert between PIL, OpenCV, bytes, base64, URLs, and files
- 🌐 **URL 支持 URL Support**: 直接从 HTTP/HTTPS URL 加载图像 Direct image loading from HTTP/HTTPS URLs
- 🎯 **类型检测 Type Detection**: 自动图像类型检测 Automatic image type detection
- 📦 **多格式支持 Multiple Formats**: 支持 JPEG、PNG、WEBP、AVIF 等格式 Support for JPEG, PNG, WEBP, AVIF, and more
- 🛡️ **错误处理 Error Handling**: 健壮的错误处理和回退机制 Robust error handling with fallback mechanisms
- 🚀 **高性能 Performance**: 针对速度和内存效率优化 Optimized for speed and memory efficiency
- 🔧 **类型提示 Type Hints**: 完整的类型注解支持 Full type annotation support
- 📝 **完整文档 Documentation**: 详细的 API 文档和示例 Comprehensive API documentation and examples

## 安装 Installation

```bash
pip install funimage
```

### 可选依赖 Optional Dependencies

安装 OpenCV 支持 For OpenCV support:
```bash
pip install funimage[opencv]
```

开发环境 For development:
```bash
pip install funimage[dev]
```

## 快速开始 Quick Start

```python
import funimage

# Convert URL to PIL Image
pil_img = funimage.convert_to_pilimg("https://example.com/image.jpg")

# Convert PIL Image to bytes
img_bytes = funimage.convert_to_bytes(pil_img)

# Convert to base64 string
base64_str = funimage.convert_to_base64_str(img_bytes)

# Save to file
funimage.convert_to_file("https://example.com/image.jpg", "output.jpg")
```

## 支持的输入类型 Supported Input Types

| Type | Description | Example |
|------|-------------|---------|
| **URL** | HTTP/HTTPS image URLs | `"https://example.com/image.jpg"` |
| **File Path** | Local file paths | `"/path/to/image.jpg"` |
| **PIL Image** | PIL Image objects | `PIL.Image.open("image.jpg")` |
| **Bytes** | Raw image bytes | `open("image.jpg", "rb").read()` |
| **Base64** | Base64 encoded strings | `"data:image/jpeg;base64,..."` |
| **NumPy Array** | OpenCV/NumPy arrays | `cv2.imread("image.jpg")` |
| **BytesIO** | BytesIO objects | `BytesIO(image_bytes)` |

## API 参考 API Reference

### 核心转换函数 Core Conversion Functions

#### `convert_to_pilimg(image, image_type=None)`
Convert any supported image format to PIL Image.

```python
# From URL
pil_img = funimage.convert_to_pilimg("https://example.com/image.jpg")

# From file
pil_img = funimage.convert_to_pilimg("/path/to/image.jpg")

# From bytes
pil_img = funimage.convert_to_pilimg(image_bytes)
```

#### `convert_to_bytes(image, image_type=None)`
Convert any supported image format to bytes.

```python
# From PIL Image
img_bytes = funimage.convert_to_bytes(pil_image)

# From URL
img_bytes = funimage.convert_to_bytes("https://example.com/image.jpg")
```

#### `convert_to_cvimg(image, image_type=None)`
Convert any supported image format to OpenCV numpy array.

```python
# From PIL Image
cv_img = funimage.convert_to_cvimg(pil_image)

# From URL
cv_img = funimage.convert_to_cvimg("https://example.com/image.jpg")
```

#### `convert_to_base64_str(image, image_type=None)`
Convert any supported image format to base64 string.

```python
# From PIL Image
b64_str = funimage.convert_to_base64_str(pil_image)

# From file
b64_str = funimage.convert_to_base64_str("/path/to/image.jpg")
```

#### `convert_to_file(image, output_path, image_type=None)`
Save any supported image format to file.

```python
# From URL to file
funimage.convert_to_file("https://example.com/image.jpg", "local_copy.jpg")

# From PIL Image to file
funimage.convert_to_file(pil_image, "output.png")
```

### 工具函数 Utility Functions

#### `parse_image_type(image, image_type=None)`
Detect the type of input image.

```python
from funimage import ImageType, parse_image_type

img_type = parse_image_type("https://example.com/image.jpg")
print(img_type)  # ImageType.URL
```

### 图像类型 Image Types

```python
from funimage import ImageType

ImageType.URL          # HTTP/HTTPS URLs
ImageType.FILE         # Local file paths  
ImageType.PIL          # PIL Image objects
ImageType.BYTES        # Raw bytes
ImageType.BASE64_STR   # Base64 strings
ImageType.NDARRAY      # NumPy arrays
ImageType.BYTESIO      # BytesIO objects
```

## 高级用法 Advanced Usage

### 显式类型指定 Explicit Type Specification

```python
from funimage import ImageType

# Explicitly specify input type
pil_img = funimage.convert_to_pilimg(
    image_data, 
    image_type=ImageType.BYTES
)
```

### 错误处理 Error Handling

```python
try:
    pil_img = funimage.convert_to_pilimg("https://invalid-url.com/image.jpg")
except Exception as e:
    print(f"Conversion failed: {e}")
```

### 批量处理 Batch Processing

```python
urls = [
    "https://example.com/image1.jpg",
    "https://example.com/image2.jpg", 
    "https://example.com/image3.jpg"
]

for i, url in enumerate(urls):
    funimage.convert_to_file(url, f"image_{i}.jpg")
```

## 示例 Examples

### 网页图像抓取 Web Scraping Images

```python
import requests
from bs4 import BeautifulSoup
import funimage

# Scrape images from a webpage
response = requests.get("https://example.com")
soup = BeautifulSoup(response.content, 'html.parser')

for i, img in enumerate(soup.find_all('img')):
    img_url = img.get('src')
    if img_url:
        try:
            funimage.convert_to_file(img_url, f"scraped_image_{i}.jpg")
            print(f"Saved image {i}")
        except Exception as e:
            print(f"Failed to save image {i}: {e}")
```

### 图像格式转换 Image Format Conversion

```python
import funimage

# Convert PNG to JPEG
png_image = funimage.convert_to_pilimg("input.png")
funimage.convert_to_file(png_image, "output.jpg")

# Convert to WebP
funimage.convert_to_file("input.jpg", "output.webp")
```

### API 集成 API Integration

```python
import funimage
import requests

def upload_image_to_api(image_path):
    # Convert image to base64 for API
    b64_str = funimage.convert_to_base64_str(image_path)
    
    payload = {
        "image": b64_str,
        "format": "jpeg"
    }
    
    response = requests.post("https://api.example.com/upload", json=payload)
    return response.json()
```

## 依赖要求 Requirements

- Python >= 3.8
- PIL/Pillow >= 9.0.0
- NumPy >= 1.20.0
- Requests >= 2.25.0

## 贡献 Contributing

欢迎贡献！请随时提交 Pull Request。Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 许可证 License

本项目基于 MIT 许可证 - 详情请查看 [LICENSE](LICENSE) 文件。This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 更新日志 Changelog

### v1.0.13
- 添加完整的类型提示 Added comprehensive type hints
- 改进错误处理和日志记录 Improved error handling and logging

## 支持 Support

如果您遇到任何问题或有疑问，请在 GitHub 上 [提交 issue](https://github.com/farfarfun/funimage/issues)。If you encounter any issues or have questions, please [open an issue](https://github.com/farfarfun/funimage/issues) on GitHub.