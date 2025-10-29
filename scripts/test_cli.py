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

            with self.assertLogs("scripts.train", level="WARNING") as log_ctx:
                train_module.main()

        mock_model.train.assert_called_once()
        called_kwargs = mock_model.train.call_args.kwargs
        self.assertIn("data", called_kwargs)
        self.assertTrue(str(called_kwargs["data"]).endswith("configs/data-initial.yaml"))
        self.assertTrue(any("预训练" in entry for entry in log_ctx.output))

    def test_missing_config_raises(self) -> None:
        argv = ["scripts/train.py", "--config", "definitely_missing.yaml"]
        with mock.patch.object(sys, "argv", argv):
            from scripts import train as train_module

            with self.assertRaises(FileNotFoundError):
                train_module.main()

    @mock.patch("scripts.train.warn_pretrained_usage")
    @mock.patch("scripts.train.torch.cuda.is_available", return_value=False)
    @mock.patch("scripts.train.YOLO")
    def test_custom_weights_no_pretrained_warning(
        self,
        mock_yolo: mock.Mock,
        _: mock.Mock,
        mock_warn: mock.Mock,
    ) -> None:
        mock_model = mock_yolo.return_value
        mock_model.train.return_value = None

        with tempfile.TemporaryDirectory() as tmpdir:
            custom_weights = Path(tmpdir) / "best.pt"
            custom_weights.write_text("dummy")

            argv = ["scripts/train.py", "--model", str(custom_weights)]
            with mock.patch.object(sys, "argv", argv):
                from scripts import train as train_module

                train_module.main()

        mock_warn.assert_not_called()


class PredictCLISmoke(TestCase):
    @mock.patch("scripts.predict.torch.cuda.is_available", return_value=False)
    @mock.patch("scripts.predict.ensure_runtime_dirs")
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
        _ensure_dirs: mock.Mock,
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

    @mock.patch("scripts.predict.torch.cuda.is_available", return_value=False)
    @mock.patch("scripts.predict.ensure_runtime_dirs")
    @mock.patch("scripts.predict.cv2.destroyAllWindows")
    @mock.patch("scripts.predict.cv2.waitKey", return_value=0)
    @mock.patch("scripts.predict.cv2.imshow")
    @mock.patch("scripts.predict.cv2.VideoWriter")
    @mock.patch("scripts.predict.cv2.VideoCapture")
    @mock.patch("scripts.predict.YOLO")
    def test_file_source_with_show_and_save(
        self,
        mock_yolo: mock.Mock,
        mock_video_capture: mock.Mock,
        mock_video_writer: mock.Mock,
        mock_imshow: mock.Mock,
        _wait_key: mock.Mock,
        mock_destroy: mock.Mock,
        _ensure_dirs: mock.Mock,
        _: mock.Mock,
    ) -> None:
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        capture_instance = mock_video_capture.return_value
        capture_instance.isOpened.return_value = True
        capture_instance.read.side_effect = [(True, frame), (False, frame)]

        track_result = SimpleNamespace(plot=lambda: frame)
        mock_model = mock_yolo.return_value
        mock_model.track.return_value = [track_result]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            weights_dir = tmp_path / "exp" / "weights"
            weights_dir.mkdir(parents=True)
            (weights_dir / "best.pt").write_text("dummy")
            output_path = tmp_path / "out.avi"

            argv = [
                "scripts/predict.py",
                "--source",
                str(tmp_path / "sample.mp4"),
                "--output",
                str(output_path),
                "--show",
            ]
            with mock.patch.object(sys, "argv", argv):
                from scripts import predict as predict_module

                with mock.patch("scripts.predict.runs_dir", return_value=tmp_path):
                    predict_module.main()

        mock_model.track.assert_called()
        mock_video_writer.assert_called_once()
        mock_imshow.assert_called()
        mock_destroy.assert_called_once()

    @mock.patch("scripts.predict.torch.cuda.is_available", return_value=False)
    @mock.patch("scripts.predict.ensure_runtime_dirs")
    @mock.patch("scripts.predict.cv2.destroyAllWindows")
    @mock.patch("scripts.predict.cv2.waitKey", return_value=0)
    @mock.patch("scripts.predict.cv2.imshow")
    @mock.patch("scripts.predict.cv2.VideoWriter")
    @mock.patch("scripts.predict.cv2.VideoCapture")
    @mock.patch("scripts.predict.YOLO")
    def test_explicit_pretrained_weights_warns(
        self,
        mock_yolo: mock.Mock,
        mock_video_capture: mock.Mock,
        _writer: mock.Mock,
        _imshow: mock.Mock,
        _wait: mock.Mock,
        _destroy: mock.Mock,
        mock_ensure_dirs: mock.Mock,
        _: mock.Mock,
    ) -> None:
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        capture_instance = mock_video_capture.return_value
        capture_instance.isOpened.return_value = True
        capture_instance.read.side_effect = [(True, frame), (False, frame)]

        track_result = SimpleNamespace(plot=lambda: frame)
        mock_yolo.return_value.track.return_value = [track_result]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            pretrained_file = tmp_path / "yolov8n.pt"
            pretrained_file.write_text("dummy")

            argv = [
                "scripts/predict.py",
                "--weights",
                str(pretrained_file),
                "--source",
                "0",
                "--no-save",
            ]
            with mock.patch.object(sys, "argv", argv):
                from scripts import predict as predict_module

                with mock.patch("scripts.predict.pretrained_dir", return_value=tmp_path):
                    with self.assertLogs("scripts.predict", level="WARNING") as log_ctx:
                        predict_module.main()

        self.assertTrue(any("预训练模型" in entry for entry in log_ctx.output))
        self.assertTrue(any("using pretrained" in entry.lower() for entry in log_ctx.output))
        mock_ensure_dirs.assert_called_once()

    @mock.patch("scripts.predict.torch.cuda.is_available", return_value=False)
    @mock.patch("scripts.predict.ensure_runtime_dirs")
    @mock.patch("scripts.predict.cv2.VideoCapture")
    @mock.patch("scripts.predict.YOLO")
    def test_missing_source_raises(
        self,
        _mock_yolo: mock.Mock,
        mock_video_capture: mock.Mock,
        _ensure_dirs: mock.Mock,
        _: mock.Mock,
    ) -> None:
        capture_instance = mock_video_capture.return_value
        capture_instance.isOpened.return_value = False

        argv = ["scripts/predict.py", "--source", "missing.mp4"]
        with mock.patch.object(sys, "argv", argv):
            from scripts import predict as predict_module

            with tempfile.TemporaryDirectory() as tmpdir:
                with mock.patch("scripts.predict.runs_dir", return_value=Path(tmpdir)):
                    with self.assertRaises(RuntimeError):
                        predict_module.main()


if __name__ == "__main__":
    main()
