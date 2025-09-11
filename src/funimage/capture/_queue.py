import queue
import threading
import time


# 自定义无缓存读视频类
class VideoCaptureQueue:
    """Customized VideoCapture, always read last frame"""

    def __init__(self, *args, **kwargs):
        import cv2

        # "camera_id" is a int type id or string name
        self.cap = cv2.VideoCapture(*args, **kwargs)
        self.q = queue.Queue(maxsize=3)
        self.stop_threads = False  # to gracefully close sub-thread
        th = threading.Thread(target=self._reader)
        th.daemon = True  # 设置工作线程为后台运行
        th.start()

    # 实时读帧，只保存最后一帧
    def _reader(self):
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

    def read(self):
        return self.q.get()

    def terminate(self):
        self.stop_threads = True
        self.cap.release()


def example():
    import cv2

    # 测试自定义VideoCapture类
    cap = VideoCaptureQueue(0)
    while True:
        ret, frame = cap.read()
        time.sleep(0.05)  # 模拟耗时操作，单位：秒
        cv2.imshow("frame", frame)
        if chr(cv2.waitKey(1) & 255) == "q":  # 按 q 退出
            cap.terminate()
            break
