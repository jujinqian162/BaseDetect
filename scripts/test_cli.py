"""Lightweight smoke tests for CLI argument handling.

These tests patch heavy dependencies (Ultralytics, OpenCV IO) so we can
verify the training and prediction entry points wire arguments correctly
without running full training or inference loops.
"""

from __future__ import annotations

import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest import TestCase, main, mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class TrainCLISmoke(TestCase):
    @mock.patch("scripts.train.torch.cuda.is_available", return_value=False)
    @mock.patch("scripts.train.YOLO")
    def test_default_dataset_runs(self, mock_yolo: mock.Mock, _: mock.Mock) -> None:
        mock_model = mock_yolo.return_value
        mock_model.train.return_value = None

        argv = ["scripts/train.py"]
        with mock.patch.object(sys, "argv", argv):
            from scripts import train as train_module

            train_module.main()

        mock_model.train.assert_called_once()
        called_kwargs = mock_model.train.call_args.kwargs
        self.assertIn("data", called_kwargs)
        self.assertTrue(str(called_kwargs["data"]).endswith("configs/data-initial.yaml"))

    @mock.patch("scripts.train.torch.cuda.is_available", return_value=False)
    @mock.patch("scripts.train.YOLO")
    def test_auto_dataset_runs(self, mock_yolo: mock.Mock, _: mock.Mock) -> None:
        mock_model = mock_yolo.return_value
        mock_model.train.return_value = None

        argv = [
            "scripts/train.py",
            "--config",
            "auto",
            "--epochs",
            "1",
            "--batch",
            "1",
            "--imgsz",
            "64",
            "--device",
            "cpu",
        ]
        with mock.patch.object(sys, "argv", argv):
            from scripts import train as train_module

            train_module.main()

        mock_model.train.assert_called_once()
        called_kwargs = mock_model.train.call_args.kwargs
        self.assertIn("data", called_kwargs)
        self.assertTrue(str(called_kwargs["data"]).endswith("datasets/demo/data.yaml"))


class PredictCLISmoke(TestCase):
    @mock.patch("scripts.predict.torch.cuda.is_available", return_value=False)
    @mock.patch("scripts.predict.cv2.destroyAllWindows")
    @mock.patch("scripts.predict.cv2.waitKey", return_value=0)
    @mock.patch("scripts.predict.cv2.imshow")
    @mock.patch("scripts.predict.cv2.VideoWriter")
    @mock.patch("scripts.predict.cv2.VideoCapture")
    @mock.patch("scripts.predict.YOLO")
    def test_camera_source_no_save(
        self,
        mock_yolo: mock.Mock,
        mock_video_capture: mock.Mock,
        _video_writer: mock.Mock,
        _imshow: mock.Mock,
        _wait_key: mock.Mock,
        _destroy: mock.Mock,
        _: mock.Mock,
    ) -> None:
        frame = np.zeros((32, 32, 3), dtype=np.uint8)

        capture_instance = mock_video_capture.return_value
        capture_instance.isOpened.return_value = True
        capture_instance.read.side_effect = [(True, frame), (False, frame)]

        track_result = SimpleNamespace(plot=lambda: frame)
        mock_model = mock_yolo.return_value
        mock_model.track.return_value = [track_result]

        argv = ["scripts/predict.py", "--source", "0", "--no-save", "--conf", "0.3"]
        with mock.patch.object(sys, "argv", argv):
            from scripts import predict as predict_module

            with tempfile.TemporaryDirectory() as tmpdir:
                with mock.patch("scripts.predict.runs_dir", return_value=Path(tmpdir)):
                    with self.assertLogs("scripts.predict", level="WARNING") as log_ctx:
                        predict_module.main()

        mock_model.track.assert_called()
        capture_instance.release.assert_called_once()
        self.assertTrue(
            any("falling back" in entry.lower() for entry in log_ctx.output),
            "Expected fallback warning missing from logs.",
        )
        self.assertTrue(
            any("警告" in entry for entry in log_ctx.output),
            "Expected bilingual warning missing from logs.",
        )


if __name__ == "__main__":
    main()
