# Changelog

## 1.0.22

### 新增

- 补充 OpenCV 转换与视频捕获队列的正常、失败和终止测试。

### 修复

- URL 图像下载统一复用 `funget`，不再维护重复的 HTTP 回退实现。

### 变更

- 构建后端迁移至 Hatchling，并提交 `uv.lock` 以保证依赖可复现。
- 日志改用 `farlog`，并将公开 API 的类型标注、docstring 和注释收敛为 Python 3.12 中文规范。

### 废弃

（无）

## 1.0.21 及更早版本

早期版本未维护 CHANGELOG，具体变更参见 git 提交历史。
