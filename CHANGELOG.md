# Changelog

## 未发布

### 修复

- `convert_url_to_bytes` 的下载失败日志不再记录完整 URL，脱敏为仅含 scheme/host/path，避免 query 中的 token 等凭据泄露到日志。
- `convert_to_cvimg` 不再用 `except Exception` 笼统捕获 OpenCV 解码失败：仅捕获 `ImportError`/`cv2.error`，且字节转换只做一次并复用结果，避免 URL/文件等有副作用的输入被重复下载或重复读取。
- 修复 `logger.error` 使用 `%s` 占位符在 `farlog`（基于 loguru）下不会被插值、消息原样输出的问题，统一改为 f-string 预格式化。

### 变更

- README 的快速开始、批量处理示例改为使用本地生成的图片，不再依赖不可验证的 `example.com` 占位地址；错误处理、网页抓取、API 集成示例改为捕获具体异常类型并以非零退出/日志记录替代裸 `except Exception` + `print`；补上缺失的 `timeout`。
- 测试用例的英文 docstring/注释统一改为中文。

## 1.0.22

### 新增

- 补充 OpenCV 转换与视频捕获队列的正常、失败和终止测试。

### 修复

- URL 图像下载统一复用 `funget`，不再维护重复的 HTTP 回退实现。

### 变更

- 构建后端迁移至 Hatchling；`uv.lock` 不再纳入版本控制。
- 日志改用 `farlog`，并将公开 API 的类型标注、docstring 和注释收敛为 Python 3.12 中文规范。

### 废弃

（无）

## 1.0.21 及更早版本

早期版本未维护 CHANGELOG，具体变更参见 git 提交历史。
