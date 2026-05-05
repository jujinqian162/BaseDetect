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
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class Coord3DUnitTests(TestCase):
    def test_scale_intrinsics_preserves_depth_across_resolution_change(self) -> None:
        from basedetect.coord3d import CameraIntrinsics, TargetSpec, pixel_to_3d, scale_intrinsics

        intrinsics = CameraIntrinsics(
            fx=1200.0,
            fy=1200.0,
            cx=960.0,
            cy=540.0,
            image_width=1920,
            image_height=1080,
        )
        target = TargetSpec(
            base_width_m=0.027,
            base_height_m=0.024,
            distance_method="average",
        )

        full_res_pos = pixel_to_3d(
            intrinsics,
            target,
            bbox_cx_px=960.0,
            bbox_cy_px=540.0,
            bbox_width_px=100.0,
            bbox_height_px=100.0,
        )

        scaled_intrinsics = scale_intrinsics(intrinsics, image_width=960, image_height=540)
        half_res_pos = pixel_to_3d(
            scaled_intrinsics,
            target,
            bbox_cx_px=480.0,
            bbox_cy_px=270.0,
            bbox_width_px=50.0,
            bbox_height_px=50.0,
        )

        self.assertAlmostEqual(scaled_intrinsics.fx, 600.0)
        self.assertAlmostEqual(scaled_intrinsics.fy, 600.0)
        self.assertAlmostEqual(scaled_intrinsics.cx, 480.0)
        self.assertAlmostEqual(scaled_intrinsics.cy, 270.0)
        self.assertAlmostEqual(half_res_pos.y, full_res_pos.y)



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
        self.assertEqual(
            Path(called_kwargs["data"]).resolve(),
            REPO_ROOT / "configs" / "data-initial.yaml",
        )
        self.assertTrue(any("没有检测到 CUDA" in entry for entry in log_ctx.output))
        self.assertTrue(any("CUDA unavailable" in entry for entry in log_ctx.output))

    def test_missing_config_raises(self) -> None:
        argv = ["scripts/train.py", "--config", "definitely_missing.yaml"]
        with mock.patch.object(sys, "argv", argv):
            from scripts import train as train_module

            with self.assertRaises(FileNotFoundError):
                train_module.main()

    @mock.patch("scripts.train.warn_cpu_training")
    @mock.patch("scripts.train.torch.cuda.is_available", return_value=True)
    @mock.patch("scripts.train.YOLO")
    def test_custom_weights_no_cpu_warning(
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
        mock_imshow: mock.Mock,
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

        argv = [
            "scripts/predict.py",
            "--source",
            "0",
            "--no-save",
            "--conf",
            "0.3",
            "--weights",
            "auto",
        ]
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
        mock_imshow.assert_called()

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
    def test_unshow_disables_display(
        self,
        mock_yolo: mock.Mock,
        mock_video_capture: mock.Mock,
        _video_writer: mock.Mock,
        mock_imshow: mock.Mock,
        _wait_key: mock.Mock,
        mock_destroy: mock.Mock,
        _ensure_dirs: mock.Mock,
        _: mock.Mock,
    ) -> None:
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        capture_instance = mock_video_capture.return_value
        capture_instance.isOpened.return_value = True
        capture_instance.read.side_effect = [(True, frame), (False, frame)]

        track_result = SimpleNamespace(plot=lambda: frame)
        mock_yolo.return_value.track.return_value = [track_result]

        argv = ["scripts/predict.py", "--source", "0", "--unshow", "--no-save"]
        with mock.patch.object(sys, "argv", argv):
            from scripts import predict as predict_module

            with tempfile.TemporaryDirectory() as tmpdir:
                with mock.patch("scripts.predict.runs_dir", return_value=Path(tmpdir)):
                    predict_module.main()

        mock_imshow.assert_not_called()
        mock_destroy.assert_not_called()


class CalibrationCLISmoke(TestCase):
    @mock.patch("scripts.calibration.resolve_image_paths")
    @mock.patch("scripts.calibration.calibrate_chessboard")
    def test_chessboard_calibration_writes_yaml(
        self,
        mock_calibrate: mock.Mock,
        mock_resolve: mock.Mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_dir = tmp_path / "images"
            input_dir.mkdir()
            image_paths = [input_dir / f"frame_{i}.jpg" for i in range(3)]
            for image_path in image_paths:
                image_path.write_text("dummy")

            output_path = tmp_path / "camera.yaml"
            mock_resolve.return_value = image_paths
            mock_calibrate.return_value = SimpleNamespace(
                rms=0.12,
                camera_matrix=np.array(
                    [[1000.0, 0.0, 960.0], [0.0, 1005.0, 540.0], [0.0, 0.0, 1.0]],
                    dtype=np.float64,
                ),
                dist_coeffs=np.array([[0.1], [-0.2], [0.001], [0.002], [0.03]], dtype=np.float64),
                image_width=1920,
                image_height=1080,
                per_view_errors=[0.11, 0.13, 0.12],
                valid_images=image_paths,
                fx=1000.0,
                fy=1005.0,
                cx=960.0,
                cy=540.0,
            )

            argv = [
                "scripts/calibration.py",
                "--input",
                str(input_dir),
                "--output",
                str(output_path),
            ]
            with mock.patch.object(sys, "argv", argv):
                from scripts import calibration as calibration_module

                calibration_module.main()

            saved = yaml.safe_load(output_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["camera"]["image_width"], 1920)
            self.assertEqual(saved["camera"]["image_height"], 1080)
            self.assertAlmostEqual(saved["camera"]["fx"], 1000.0)
            self.assertAlmostEqual(saved["camera"]["fy"], 1005.0)
            self.assertEqual(len(saved["camera"]["dist_coeffs"]), 5)
            mock_calibrate.assert_called_once()


class CalibrationVideoCLISmoke(TestCase):
    @mock.patch("scripts.calibration_video.cv2.VideoCapture")
    @mock.patch("scripts.calibration_video.cv2.calibrateCamera")
    @mock.patch("scripts.calibration_video.compute_reprojection_errors", return_value=[0.2, 0.2, 0.2])
    def test_video_calibration_writes_report_and_yaml(
        self,
        _mock_reprojection: mock.Mock,
        mock_calibrate_camera: mock.Mock,
        mock_video_capture: mock.Mock,
    ) -> None:
        frame = np.zeros((80, 120, 3), dtype=np.uint8)
        capture_instance = mock_video_capture.return_value
        capture_instance.isOpened.return_value = True
        capture_instance.read.side_effect = [(True, frame), (True, frame), (True, frame), (False, frame)]
        capture_instance.get.return_value = 100.0

        fake_corners = np.zeros((54, 1, 2), dtype=np.float32)
        mock_calibrate_camera.return_value = (
            0.2,
            np.array([[900.0, 0.0, 60.0], [0.0, 910.0, 40.0], [0.0, 0.0, 1.0]], dtype=np.float64),
            np.array([[0.01], [-0.02], [0.001], [0.002], [0.03]], dtype=np.float64),
            [np.zeros((3, 1), dtype=np.float64)] * 3,
            [np.zeros((3, 1), dtype=np.float64)] * 3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            video_path = tmp_path / "sample.mp4"
            video_path.write_text("dummy")
            output_path = tmp_path / "camera.yaml"
            report_path = tmp_path / "report.csv"

            argv = [
                "scripts/calibration_video.py",
                "--input",
                str(video_path),
                "--output",
                str(output_path),
                "--report",
                str(report_path),
                "--pattern",
                "chessboard",
                "--max-frames",
                "3",
            ]
            with mock.patch.object(sys, "argv", argv):
                with mock.patch("scripts.calibration_video.detect_chessboard_in_frame") as mock_detect:
                    from scripts import calibration_video as calibration_video_module

                    mock_detect.side_effect = [
                        (
                            fake_corners,
                            calibration_video_module.FrameRecord(
                                frame_index=0,
                                timestamp_ms=0.0,
                                success=True,
                                pattern="chessboard",
                                image_width=120,
                                image_height=80,
                                points_count=54,
                                marker_count=0,
                                reason="ok",
                            ),
                        ),
                        (
                            fake_corners,
                            calibration_video_module.FrameRecord(
                                frame_index=1,
                                timestamp_ms=0.0,
                                success=True,
                                pattern="chessboard",
                                image_width=120,
                                image_height=80,
                                points_count=54,
                                marker_count=0,
                                reason="ok",
                            ),
                        ),
                        (
                            fake_corners,
                            calibration_video_module.FrameRecord(
                                frame_index=2,
                                timestamp_ms=0.0,
                                success=True,
                                pattern="chessboard",
                                image_width=120,
                                image_height=80,
                                points_count=54,
                                marker_count=0,
                                reason="ok",
                            ),
                        ),
                    ]

                    calibration_video_module.main()

            saved = yaml.safe_load(output_path.read_text(encoding="utf-8"))
            report_text = report_path.read_text(encoding="utf-8")
            self.assertAlmostEqual(saved["camera"]["fx"], 900.0)
            self.assertAlmostEqual(saved["camera"]["fy"], 910.0)
            self.assertIn("frame_index", report_text)
            self.assertIn("ok", report_text)
            self.assertIn("reprojection_error", report_text)
            self.assertIn("used_in_calibration", report_text)
            self.assertIn("0.2", report_text)
            self.assertGreaterEqual(report_text.count("True"), 1)
            capture_instance.release.assert_called_once()


class PredictCLIAdditionalSmoke(TestCase):

    @mock.patch("scripts.predict.torch.cuda.is_available", return_value=False)
    @mock.patch("scripts.predict.ensure_runtime_dirs")
    @mock.patch("scripts.predict.cv2.destroyAllWindows")
    @mock.patch("scripts.predict.cv2.waitKey", return_value=0)
    @mock.patch("scripts.predict.cv2.imshow")
    @mock.patch("scripts.predict.cv2.VideoWriter")
    @mock.patch("scripts.predict.cv2.VideoCapture")
    @mock.patch("scripts.predict._create_apriltag_detector")
    @mock.patch("scripts.predict.load_camera_config")
    @mock.patch("scripts.predict.load_apriltag_config")
    @mock.patch("scripts.predict.YOLO")
    def test_apriltag_mode_logs_pose_with_base_coords(
        self,
        mock_yolo: mock.Mock,
        mock_load_apriltag_config: mock.Mock,
        mock_load_camera_config: mock.Mock,
        mock_create_apriltag_detector: mock.Mock,
        mock_video_capture: mock.Mock,
        _writer: mock.Mock,
        _imshow: mock.Mock,
        _wait: mock.Mock,
        _destroy: mock.Mock,
        _ensure_dirs: mock.Mock,
        _: mock.Mock,
    ) -> None:
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        capture_instance = mock_video_capture.return_value
        capture_instance.isOpened.return_value = True
        capture_instance.read.side_effect = [(True, frame), (False, frame)]

        track_result = SimpleNamespace(plot=lambda: frame)
        boxes = SimpleNamespace(
            xyxy=np.array([[8.0, 8.0, 24.0, 24.0]], dtype=np.float32),
            id=np.array([3], dtype=np.int32),
            conf=np.array([0.88], dtype=np.float32),
        )
        track_result.boxes = boxes
        mock_yolo.return_value.track.return_value = [track_result]

        detection = {
            "id": 7,
            "center": np.array([16.0, 18.0], dtype=np.float32),
            "lb-rb-rt-lt": np.array(
                [[10.0, 22.0], [22.0, 22.0], [22.0, 10.0], [10.0, 10.0]],
                dtype=np.float32,
            ),
            "margin": 55.0,
        }
        detector = mock.Mock()
        detector.detect.return_value = [detection]
        detector.estimate_tag_pose.return_value = {
            "t": np.array([[0.1], [0.2], [1.5]], dtype=np.float32),
            "error": 0.0123,
        }
        mock_load_apriltag_config.return_value = SimpleNamespace(
            family="tag36h11",
            tag_size_m=0.05,
            mirror_input=False,
        )
        mock_load_camera_config.return_value = (
            SimpleNamespace(
                fx=657.13299,
                fy=657.004538,
                cx=301.772867,
                cy=253.594519,
                image_width=32,
                image_height=32,
            ),
            SimpleNamespace(base_width_m=0.029, base_height_m=0.026, distance_method="average"),
        )
        mock_create_apriltag_detector.return_value = detector

        argv = [
            "scripts/predict.py",
            "--type",
            "base_coord",
            "--source",
            "fake.mp4",
            "--no-save",
            "--unshow",
            "--apriltag",
        ]
        with mock.patch.object(sys, "argv", argv):
            from scripts import predict as predict_module

            with self.assertLogs("scripts.predict", level="INFO") as log_ctx:
                predict_module.main()

        mock_yolo.assert_called_once()
        mock_load_apriltag_config.assert_called_once()
        mock_yolo.return_value.track.assert_called_once()
        detector.detect.assert_called_once()
        detector.estimate_tag_pose.assert_called_once()
        self.assertTrue(any("track=#3" in entry for entry in log_ctx.output))
        self.assertTrue(any("tag_id=7" in entry for entry in log_ctx.output))
        self.assertTrue(any("X=+0.100m Y=1.500m Z=-0.200m" in entry for entry in log_ctx.output))

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


class PredictConfigModeSmoke(TestCase):
    @mock.patch("scripts.predict.torch.cuda.is_available", return_value=False)
    @mock.patch("scripts.predict.ensure_runtime_dirs")
    @mock.patch("scripts.predict.cv2.destroyAllWindows")
    @mock.patch("scripts.predict.cv2.waitKey", return_value=0)
    @mock.patch("scripts.predict.cv2.imshow")
    @mock.patch("scripts.predict.cv2.putText")
    @mock.patch("scripts.predict.cv2.VideoWriter")
    @mock.patch("scripts.predict.cv2.VideoCapture")
    @mock.patch("scripts.predict.load_camera_config")
    @mock.patch("scripts.predict.YOLO")
    def test_status_mode_from_yaml_draws_status_without_coord3d(
        self,
        mock_yolo: mock.Mock,
        mock_load_camera_config: mock.Mock,
        mock_video_capture: mock.Mock,
        _writer: mock.Mock,
        _put_text: mock.Mock,
        _imshow: mock.Mock,
        _wait: mock.Mock,
        _destroy: mock.Mock,
        _ensure_dirs: mock.Mock,
        _: mock.Mock,
    ) -> None:
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        capture_instance = mock_video_capture.return_value
        capture_instance.isOpened.return_value = True
        capture_instance.read.side_effect = [(True, frame), (False, frame)]

        track_result = SimpleNamespace(plot=lambda: frame)
        track_result.boxes = SimpleNamespace(
            xyxy=np.array([[1.0, 1.0, 10.0, 10.0], [12.0, 1.0, 20.0, 10.0]], dtype=np.float32),
            cls=np.array([0.0, 2.0], dtype=np.float32),
        )
        track_result.names = {0: "spearhead", 1: "palm", 2: "fist"}
        mock_yolo.return_value.track.return_value = [track_result]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "predict.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "predict": {
                            "type": "status",
                            "source": "0",
                            "save": False,
                            "show": False,
                            "weights": "auto",
                        }
                    },
                    sort_keys=False,
                    allow_unicode=True,
                ),
                encoding="utf-8",
            )

            argv = ["scripts/predict.py", "--config", str(config_path)]
            with mock.patch.object(sys, "argv", argv):
                from scripts import predict as predict_module

                with self.assertLogs("scripts.predict", level="INFO") as log_ctx:
                    predict_module.main()

        mock_load_camera_config.assert_not_called()
        self.assertTrue(any("status=['spearhead', 'fist']" in entry for entry in log_ctx.output))


if __name__ == "__main__":
    main()
