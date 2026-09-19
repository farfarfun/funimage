import queue
import threading
from typing import Any


class VideoCaptureQueue:
    """在后台持续读取视频，队列中只保留最新帧。

    Args:
        *args: 传给 `cv2.VideoCapture` 的位置参数。
        **kwargs: 传给 `cv2.VideoCapture` 的关键字参数。
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        import cv2

        self.cap = cv2.VideoCapture(*args, **kwargs)
        self.q: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=3)
        self.stop_threads = False
        th = threading.Thread(target=self._reader)
        th.daemon = True
        th.start()

    def _reader(self) -> None:
        while not self.stop_threads:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put((ret, frame))

    def read(self) -> tuple[bool, Any]:
        """等待并返回最新的 `(是否成功, 图像帧)`。"""
        return self.q.get()

    def terminate(self) -> None:
        """停止后台读取并释放视频捕获器。"""
        self.stop_threads = True
        self.cap.release()
