import queue
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

from funimage.capture import VideoCaptureQueue


class TestVideoCaptureQueue:
    @patch("funimage.capture._queue.threading.Thread")
    def test_initializes_and_terminates_capture(self, mock_thread):
        """测试初始化时启动后台读取线程，终止时释放底层采集器。"""
        backend = Mock()
        cv2 = SimpleNamespace(VideoCapture=Mock(return_value=backend))

        with patch.dict(sys.modules, {"cv2": cv2}):
            capture = VideoCaptureQueue(0)

        cv2.VideoCapture.assert_called_once_with(0)
        mock_thread.assert_called_once_with(target=capture._reader)
        assert mock_thread.return_value.daemon is True
        mock_thread.return_value.start.assert_called_once()

        capture.terminate()
        assert capture.stop_threads is True
        backend.release.assert_called_once()

    def test_reader_keeps_only_latest_frame(self):
        """测试后台读取循环只在队列中保留最新一帧。"""
        capture = object.__new__(VideoCaptureQueue)
        capture.cap = Mock()
        capture.cap.read.side_effect = [
            (True, "first"),
            (True, "second"),
            (False, None),
        ]
        capture.q = queue.Queue(maxsize=3)
        capture.stop_threads = False

        capture._reader()

        assert capture.read() == (True, "second")
        assert capture.q.empty()

    def test_reader_stops_when_capture_fails(self):
        """测试采集失败（读取返回 False）时后台读取循环会退出。"""
        capture = object.__new__(VideoCaptureQueue)
        capture.cap = Mock()
        capture.cap.read.return_value = (False, None)
        capture.q = queue.Queue(maxsize=3)
        capture.stop_threads = False

        capture._reader()

        assert capture.q.empty()
