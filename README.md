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

PyPI 当前发布的最新版本为 `1.0.19`。本仓库的 `1.0.22` 仍未发布；需要使用当前源码时，在检出仓库根目录执行：

```bash
pip install .
```

### 可选依赖 Optional Dependencies

安装 OpenCV 支持 For OpenCV support:
```bash
pip install funimage[opencv]
```

安装网页抓取和 API 集成示例所需依赖 For the web scraping and API integration examples:
```bash
pip install funimage[web]
```

## 快速开始 Quick Start

下面示例只依赖 PIL 在本地生成的图像，无需网络即可直接运行 The example below only uses a PIL-generated local image and runs without any network access:

```python
import PIL.Image

import funimage

# 本地生成一张测试图片，无需网络 Generate a local test image, no network required
pil_img = PIL.Image.new("RGB", (64, 64), color="red")

# Convert PIL Image to bytes
img_bytes = funimage.convert_to_bytes(pil_img)

# Convert to base64 string
base64_str = funimage.convert_to_base64_str(img_bytes)

# Save to file
funimage.convert_to_file(pil_img, "output.jpg")
```

> 从 HTTP/HTTPS URL 加载图像需要目标地址真实可达；地址不可达时 `convert_to_pilimg`、
> `convert_to_file` 等函数会抛出 `ValueError`，调用方应按具体异常类型捕获处理
> （见下方「错误处理」）。Loading from an HTTP/HTTPS URL requires the target address to
> be reachable; on failure, functions like `convert_to_pilimg` and `convert_to_file`
> raise `ValueError`, which callers should catch by specific exception type (see
> "Error Handling" below).

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

`convert_url_to_bytes` 在 URL 下载失败时返回 `None`。其他以 URL 为输入的公开转换函数（如
`convert_to_pilimg`、`convert_to_bytes` 和 `convert_to_file`）会将该失败转换为 `ValueError`；
不受支持的输入同样会抛出 `ValueError`。应按具体异常类型捕获并以非零退出码结束，而不是用
`except Exception` 吞掉错误。

`convert_url_to_bytes` returns `None` when a URL download fails. Public conversion
functions that accept URL input, including `convert_to_pilimg`, `convert_to_bytes`,
and `convert_to_file`, convert that failure to `ValueError`; unsupported input also
raises `ValueError`. Catch that specific exception type and exit with a non-zero status
instead of swallowing errors with `except Exception`:

```python
import sys

try:
    pil_img = funimage.convert_to_pilimg("https://invalid-url.example/image.jpg")
except ValueError as exc:
    sys.exit(f"图像转换失败：{exc}")
```

### 批量处理 Batch Processing

```python
import PIL.Image

images = [
    PIL.Image.new("RGB", (32, 32), color="red"),
    PIL.Image.new("RGB", (32, 32), color="green"),
    PIL.Image.new("RGB", (32, 32), color="blue"),
]

for i, image in enumerate(images):
    funimage.convert_to_file(image, f"image_{i}.jpg")
```

## 示例 Examples

### 网页图像抓取 Web Scraping Images

该示例依赖目标网页可访问，运行前请确认网络环境 This example requires the target
page to be reachable; make sure network access is available before running it:

```bash
pip install funimage[web]
```

```python
from farlog import getLogger
import requests
from bs4 import BeautifulSoup
import funimage

logger = getLogger("funimage-example")

# Scrape images from a webpage
response = requests.get("https://example.com", timeout=10)
soup = BeautifulSoup(response.content, 'html.parser')

for i, img in enumerate(soup.find_all('img')):
    img_url = img.get('src')
    if img_url:
        try:
            funimage.convert_to_file(img_url, f"scraped_image_{i}.jpg")
        except ValueError as exc:
            logger.error("第 {} 张图片保存失败：{}", i, exc)
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

该示例需要 `requests`，请先安装可选的 `web` 依赖 This example requires `requests`; install the optional `web` dependency first:

```bash
pip install funimage[web]
```

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
    
    response = requests.post("https://api.example.com/upload", json=payload, timeout=10)
    return response.json()
```

## 依赖要求 Requirements

- Python >= 3.12
- PIL/Pillow >= 9.0.0
- NumPy >= 1.20.0
- FunGet >= 1.1.63

## 贡献 Contributing

欢迎贡献！请随时提交 Pull Request。Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork 本仓库 Fork the repository
2. 新建特性分支 Create your feature branch (`git checkout -b feature/avif-support`)
3. 提交改动 Commit your changes (`git commit -m 'feat: 新增 AVIF 图像转换支持'`)
4. 推送分支 Push to the branch (`git push origin feature/avif-support`)
5. 发起 Pull Request Open a Pull Request

提交信息用中文，格式为 `<类型>: <做了什么>`，类型取 `feat`/`fix`/`docs`/`refactor`/`test`/`chore`；
正文说明「为什么这么改」而不是复述 diff，一次提交只做一件事。

## 支持 Support

如果您遇到任何问题或有疑问，请在 GitHub 上 [提交 issue](https://github.com/farfarfun/funimage/issues)。If you encounter any issues or have questions, please [open an issue](https://github.com/farfarfun/funimage/issues) on GitHub.

---

## 关于 farfarfun

[farfarfun](https://github.com/farfarfun) 是一个专注于实用工具库的开源组织，
涵盖云存储、数据处理、AI、多媒体与开发工具链等方向。

- 🏠 组织主页：<https://github.com/farfarfun>
- 📦 PyPI：<https://pypi.org/user/niuliangtao/>
- 📧 联系：farfarfun@qq.com

本项目基于 [MIT](LICENSE) 协议开源。
