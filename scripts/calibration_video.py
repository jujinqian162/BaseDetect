"""Video-driven camera calibration helper.

This script samples frames from a video, tries to detect a chessboard or
ChArUco target on each frame, records per-frame detection metadata to CSV,
then calibrates the camera from all valid detections.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from basedetect.paths import artifacts_dir, camera_config
from scripts.calibration import (  # noqa: E402
	ARUCO,
	CalibrationResult,
	build_chessboard_object_points,
	compute_reprojection_errors,
	print_summary,
	update_camera_yaml,
)


LOGGER = logging.getLogger("scripts.calibration_video")


@dataclass
class FrameRecord:
	frame_index: int
	timestamp_ms: float
	success: bool
	pattern: str
	image_width: int
	image_height: int
	points_count: int
	marker_count: int
	reason: str
	reprojection_error: float | None = None
	used_in_calibration: bool = False


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Calibrate camera intrinsics directly from a video sequence."
	)
	parser.add_argument("--input", type=Path, required=True, help="Input video path.")
	parser.add_argument(
		"--pattern",
		choices=("chessboard", "charuco"),
		default="charuco",
		help="Calibration target type.",
	)
	parser.add_argument("--cols", type=int, default=6, help="Pattern columns.")
	parser.add_argument("--rows", type=int, default=6, help="Pattern rows.")
	parser.add_argument("--square-size", type=float, default=30.0, help="Square size in millimeters.")
	parser.add_argument("--marker-size", type=float, default=23.0, help="Marker size in millimeters for ChArUco.")
	parser.add_argument("--aruco-dict", default="DICT_4X4_50", help="ArUco dictionary name for ChArUco mode.")
	parser.add_argument(
		"--frame-step",
		type=int,
		default=1,
		help="Process every Nth frame. Increase for faster sampling.",
	)
	parser.add_argument(
		"--max-frames",
		type=int,
		default=None,
		help="Optional cap on processed frames after frame-step filtering.",
	)
	parser.add_argument(
		"--max-reprojection-error",
		type=float,
		default=1.0,
		help="Drop successful frames whose per-frame reprojection error exceeds this threshold, then recalibrate.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default="configs-temp/camera.yaml",
		help="YAML file to receive calibration results.",
	)
	parser.add_argument(
		"--report",
		type=Path,
		default=artifacts_dir() / "outputs" / "calibration_video_report.csv",
		help="CSV file that records every processed frame.",
	)
	parser.add_argument(
		"--save-debug",
		type=Path,
		default=None,
		help="Optional folder to save successful detection previews.",
	)
	parser.add_argument("--show", action="store_true", help="Preview detections while processing.")
	return parser


def _termination_criteria() -> tuple[int, int, float]:
	return (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)


def save_debug_frame(
	image: np.ndarray,
	frame_index: int,
	save_debug_dir: Path | None,
) -> None:
	if save_debug_dir is None:
		return
	save_debug_dir.mkdir(parents=True, exist_ok=True)
	output_path = save_debug_dir / f"frame_{frame_index:06d}.jpg"
	cv2.imwrite(str(output_path), image)


def maybe_show_preview(image: np.ndarray, show: bool) -> None:
	if not show:
		return
	cv2.imshow("calibration-video-preview", image)
	cv2.waitKey(1)


def detect_chessboard_in_frame(
	frame: np.ndarray,
	cols: int,
	rows: int,
	frame_index: int,
	save_debug_dir: Path | None,
	show: bool,
) -> tuple[np.ndarray | None, FrameRecord]:
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
	found, corners = cv2.findChessboardCorners(gray, (cols, rows), None, flags)
	if not found or corners is None:
		record = FrameRecord(
			frame_index=frame_index,
			timestamp_ms=0.0,
			success=False,
			pattern="chessboard",
			image_width=frame.shape[1],
			image_height=frame.shape[0],
			points_count=0,
			marker_count=0,
			reason="corners_not_found",
		)
		return None, record

	refined = cv2.cornerSubPix(
		gray,
		corners,
		winSize=(11, 11),
		zeroZone=(-1, -1),
		criteria=_termination_criteria(),
	)
	preview = frame.copy()
	cv2.drawChessboardCorners(preview, (cols, rows), refined, True)
	save_debug_frame(preview, frame_index, save_debug_dir)
	maybe_show_preview(preview, show)
	record = FrameRecord(
		frame_index=frame_index,
		timestamp_ms=0.0,
		success=True,
		pattern="chessboard",
		image_width=frame.shape[1],
		image_height=frame.shape[0],
		points_count=int(len(refined)),
		marker_count=0,
		reason="ok",
	)
	return refined, record


def detect_charuco_in_frame(
	frame: np.ndarray,
	cols: int,
	rows: int,
	square_size: float,
	marker_size: float,
	aruco_dict_name: str,
	frame_index: int,
	save_debug_dir: Path | None,
	show: bool,
) -> tuple[np.ndarray | None, np.ndarray | None, FrameRecord]:
	if ARUCO is None:
		raise RuntimeError(
			"Current OpenCV build does not include cv2.aruco. Install opencv-contrib-python or use --pattern chessboard."
		)

	aruco_module = ARUCO
	dictionary_id = getattr(aruco_module, aruco_dict_name, None)
	if dictionary_id is None:
		raise ValueError(f"Unsupported ArUco dictionary: {aruco_dict_name}")

	dictionary = aruco_module.getPredefinedDictionary(dictionary_id)
	board = aruco_module.CharucoBoard((cols, rows), square_size, marker_size, dictionary)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	detector = aruco_module.ArucoDetector(dictionary, aruco_module.DetectorParameters())
	marker_corners, marker_ids, _ = detector.detectMarkers(gray)
	if marker_ids is None or len(marker_ids) == 0:
		record = FrameRecord(
			frame_index=frame_index,
			timestamp_ms=0.0,
			success=False,
			pattern="charuco",
			image_width=frame.shape[1],
			image_height=frame.shape[0],
			points_count=0,
			marker_count=0,
			reason="markers_not_found",
		)
		return None, None, record

	interpolate = getattr(aruco_module, "interpolateCornersCharuco", None)
	if interpolate is None:
		raise RuntimeError(
			"Current OpenCV build is missing interpolateCornersCharuco. Install opencv-contrib-python or use --pattern chessboard."
		)

	charuco_ret, charuco_corners, charuco_ids = interpolate(marker_corners, marker_ids, gray, board)
	if not charuco_ret or charuco_corners is None or charuco_ids is None or len(charuco_ids) < 4:
		record = FrameRecord(
			frame_index=frame_index,
			timestamp_ms=0.0,
			success=False,
			pattern="charuco",
			image_width=frame.shape[1],
			image_height=frame.shape[0],
			points_count=0,
			marker_count=int(len(marker_ids)),
			reason="insufficient_charuco_corners",
		)
		return None, None, record

	preview = frame.copy()
	aruco_module.drawDetectedMarkers(preview, marker_corners, marker_ids)
	aruco_module.drawDetectedCornersCharuco(preview, charuco_corners, charuco_ids)
	save_debug_frame(preview, frame_index, save_debug_dir)
	maybe_show_preview(preview, show)
	record = FrameRecord(
		frame_index=frame_index,
		timestamp_ms=0.0,
		success=True,
		pattern="charuco",
		image_width=frame.shape[1],
		image_height=frame.shape[0],
		points_count=int(len(charuco_ids)),
		marker_count=int(len(marker_ids)),
		reason="ok",
	)
	return charuco_corners, charuco_ids, record


def write_report(report_path: Path, records: list[FrameRecord]) -> None:
	report_path.parent.mkdir(parents=True, exist_ok=True)
	with report_path.open("w", encoding="utf-8", newline="") as file:
		writer = csv.DictWriter(file, fieldnames=list(FrameRecord.__annotations__.keys()))
		writer.writeheader()
		for record in records:
			writer.writerow(asdict(record))


def calibrate_chessboard_dataset(
	object_points: list[np.ndarray],
	image_points: list[np.ndarray],
	image_size: tuple[int, int],
	valid_frame_names: list[Path],
) -> tuple[CalibrationResult, list[float]]:
	initial_camera_matrix = np.eye(3, dtype=np.float64)
	initial_dist_coeffs = np.zeros((8, 1), dtype=np.float64)
	rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
		object_points,
		image_points,
		image_size,
		initial_camera_matrix,
		initial_dist_coeffs,
	)
	per_view_errors = compute_reprojection_errors(
		object_points, image_points, rvecs, tvecs, camera_matrix, dist_coeffs
	)
	result = CalibrationResult(
		rms=float(rms),
		camera_matrix=camera_matrix,
		dist_coeffs=dist_coeffs,
		image_width=image_size[0],
		image_height=image_size[1],
		per_view_errors=per_view_errors,
		valid_images=valid_frame_names,
	)
	return result, per_view_errors


def recalibrate_chessboard_after_filtering(
	object_points: list[np.ndarray],
	image_points: list[np.ndarray],
	image_size: tuple[int, int],
	valid_frame_names: list[Path],
	success_record_indices: list[int],
	records: list[FrameRecord],
	max_reprojection_error: float,
) -> CalibrationResult:
	initial_result, per_view_errors = calibrate_chessboard_dataset(
		object_points,
		image_points,
		image_size,
		valid_frame_names,
	)

	for record in records:
		record.used_in_calibration = False
		if record.success:
			record.reprojection_error = None

	for record_index, error in zip(success_record_indices, per_view_errors):
		records[record_index].reprojection_error = float(error)

	kept_indices = [
		idx for idx, error in enumerate(per_view_errors) if float(error) <= max_reprojection_error
	]
	if len(kept_indices) < 3:
		LOGGER.warning(
			"Filtering by reprojection error <= %.3f leaves only %d frames; keeping original calibration result.",
			max_reprojection_error,
			len(kept_indices),
		)
		for record_index in success_record_indices:
			records[record_index].used_in_calibration = True
		return initial_result

	if len(kept_indices) == len(per_view_errors):
		for record_index in success_record_indices:
			records[record_index].used_in_calibration = True
		LOGGER.info("No successful frames exceeded reprojection error threshold %.3f.", max_reprojection_error)
		return initial_result

	LOGGER.info(
		"Filtering out %d/%d successful chessboard frames above reprojection error %.3f and recalibrating.",
		len(per_view_errors) - len(kept_indices),
		len(per_view_errors),
		max_reprojection_error,
	)

	filtered_object_points = [object_points[idx] for idx in kept_indices]
	filtered_image_points = [image_points[idx] for idx in kept_indices]
	filtered_frame_names = [valid_frame_names[idx] for idx in kept_indices]
	filtered_record_indices = [success_record_indices[idx] for idx in kept_indices]

	final_result, final_errors = calibrate_chessboard_dataset(
		filtered_object_points,
		filtered_image_points,
		image_size,
		filtered_frame_names,
	)
	for record_index, error in zip(filtered_record_indices, final_errors):
		records[record_index].reprojection_error = float(error)
		records[record_index].used_in_calibration = True
	return final_result


def calibrate_from_video_frames(
	video_path: Path,
	pattern: str,
	cols: int,
	rows: int,
	square_size: float,
	marker_size: float,
	aruco_dict_name: str,
	frame_step: int,
	max_frames: int | None,
	max_reprojection_error: float,
	save_debug_dir: Path | None,
	show: bool,
) -> tuple[CalibrationResult, list[FrameRecord]]:
	if frame_step <= 0:
		raise ValueError("--frame-step must be positive.")

	capture = cv2.VideoCapture(str(video_path))
	if not capture.isOpened():
		raise RuntimeError(f"Unable to open video: {video_path}")

	records: list[FrameRecord] = []
	image_size: tuple[int, int] | None = None
	processed_frames = 0
	frame_index = -1

	object_points: list[np.ndarray] = []
	image_points: list[np.ndarray] = []
	charuco_corners_list: list[np.ndarray] = []
	charuco_ids_list: list[np.ndarray] = []
	valid_frame_names: list[Path] = []
	success_record_indices: list[int] = []

	if pattern == "chessboard":
		template_object_points = build_chessboard_object_points(cols, rows, square_size)
	else:
		template_object_points = None

	try:
		while True:
			ok, frame = capture.read()
			if not ok:
				LOGGER.info(
					"Frame reading finished at raw frame_index=%d, processed_frames=%d.",
					frame_index,
					processed_frames,
				)
				break

			frame_index += 1
			
			if frame_index % frame_step != 0:
				continue
			processed_frames += 1
			if processed_frames % 100 == 0:
				print(f"\rprocessed_frame {processed_frames}")
			
			if max_frames is not None and processed_frames > max_frames:
				break

			current_size = (frame.shape[1], frame.shape[0])
			if image_size is None:
				image_size = current_size
			elif image_size != current_size:
				raise ValueError(
					f"Video frames changed resolution from {image_size} to {current_size}; calibration requires a fixed size."
				)

			timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
			frame_label = Path(f"frame_{frame_index:06d}.jpg")

			if pattern == "chessboard":
				corners, record = detect_chessboard_in_frame(
					frame,
					cols,
					rows,
					frame_index,
					save_debug_dir,
					show,
				)
				record.timestamp_ms = timestamp_ms
				records.append(record)
				record_index = len(records) - 1
				if corners is None or template_object_points is None:
					continue
				object_points.append(template_object_points.copy())
				image_points.append(corners)
				valid_frame_names.append(frame_label)
				success_record_indices.append(record_index)
			else:
				charuco_corners, charuco_ids, record = detect_charuco_in_frame(
					frame,
					cols,
					rows,
					square_size,
					marker_size,
					aruco_dict_name,
					frame_index,
					save_debug_dir,
					show,
				)
				record.timestamp_ms = timestamp_ms
				records.append(record)
				record_index = len(records) - 1
				if charuco_corners is None or charuco_ids is None:
					continue
				charuco_corners_list.append(charuco_corners)
				charuco_ids_list.append(charuco_ids)
				valid_frame_names.append(frame_label)
				success_record_indices.append(record_index)
	finally:
		capture.release()
		if show:
			cv2.destroyAllWindows()

	LOGGER.info(
		"Video scan complete. pattern=%s processed_frames=%d successful_frames=%d records=%d",
		pattern,
		processed_frames,
		len(success_record_indices),
		len(records),
	)

	if image_size is None:
		raise RuntimeError("No frames were processed from the video.")

	if pattern == "chessboard":
		if len(image_points) < 3:
			raise RuntimeError("Need at least 3 successful frames with chessboard corners for calibration.")
		LOGGER.info(
			"Starting chessboard calibration with %d successful frames and image_size=%s.",
			len(image_points),
			image_size,
		)
		result = recalibrate_chessboard_after_filtering(
			object_points,
			image_points,
			image_size,
			valid_frame_names,
			success_record_indices,
			records,
			max_reprojection_error,
		)
		LOGGER.info("Chessboard calibration finished. final rms=%.6f", result.rms)
		return result, records

	if len(charuco_corners_list) < 3:
		raise RuntimeError("Need at least 3 successful frames with ChArUco corners for calibration.")
	LOGGER.info(
		"Starting ChArUco calibration with %d successful frames and image_size=%s.",
		len(charuco_corners_list),
		image_size,
	)
	if ARUCO is None:
		raise RuntimeError(
			"Current OpenCV build does not include cv2.aruco. Install opencv-contrib-python or use --pattern chessboard."
		)

	aruco_module = ARUCO
	dictionary_id = getattr(aruco_module, aruco_dict_name, None)
	if dictionary_id is None:
		raise ValueError(f"Unsupported ArUco dictionary: {aruco_dict_name}")
	dictionary = aruco_module.getPredefinedDictionary(dictionary_id)
	board = aruco_module.CharucoBoard((cols, rows), square_size, marker_size, dictionary)
	calibrate_charuco = getattr(aruco_module, "calibrateCameraCharuco", None)
	if calibrate_charuco is None:
		raise RuntimeError(
			"Current OpenCV build is missing calibrateCameraCharuco. Install opencv-contrib-python or use --pattern chessboard."
		)

	initial_camera_matrix = np.eye(3, dtype=np.float64)
	initial_dist_coeffs = np.zeros((8, 1), dtype=np.float64)
	rms, camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_charuco(
		charuco_corners_list,
		charuco_ids_list,
		board,
		image_size,
		initial_camera_matrix,
		initial_dist_coeffs,
	)

	object_points = []
	image_points = []
	for corners, ids in zip(charuco_corners_list, charuco_ids_list):
		board_points = np.asarray(board.getChessboardCorners(), dtype=np.float32)[ids.flatten().tolist()]
		object_points.append(board_points.astype(np.float32))
		image_points.append(corners.astype(np.float32))
	per_view_errors = compute_reprojection_errors(
		object_points, image_points, rvecs, tvecs, camera_matrix, dist_coeffs
	)
	LOGGER.info("ChArUco calibration finished. rms=%.6f", float(rms))
	for record_index, error in zip(success_record_indices, per_view_errors):
		records[record_index].reprojection_error = float(error)
		records[record_index].used_in_calibration = True
	result = CalibrationResult(
		rms=float(rms),
		camera_matrix=camera_matrix,
		dist_coeffs=dist_coeffs,
		image_width=image_size[0],
		image_height=image_size[1],
		per_view_errors=per_view_errors,
		valid_images=valid_frame_names,
	)
	return result, records


def main() -> None:
	logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
	args = build_parser().parse_args()

	if args.square_size <= 0:
		raise ValueError("--square-size must be positive.")
	if args.pattern == "charuco" and args.marker_size <= 0:
		raise ValueError("--marker-size must be positive.")
	if not args.input.exists():
		raise FileNotFoundError(f"Video file not found: {args.input}")

	result, records = calibrate_from_video_frames(
		video_path=args.input,
		pattern=args.pattern,
		cols=args.cols,
		rows=args.rows,
		square_size=args.square_size,
		marker_size=args.marker_size,
		aruco_dict_name=args.aruco_dict,
		frame_step=args.frame_step,
		max_frames=args.max_frames,
		max_reprojection_error=args.max_reprojection_error,
		save_debug_dir=args.save_debug,
		show=args.show,
	)
	LOGGER.info("Writing calibration report to %s", args.report)
	write_report(args.report, records)
	LOGGER.info("Writing camera parameters to %s", args.output)
	update_camera_yaml(args.output, result)
	print_summary(result, args.output)
	print(f"  report csv : {args.report}")
	print(f"  frames seen: {len(records)}")
	print(f"  frames used: {sum(1 for record in records if record.success)}")


if __name__ == "__main__":
	main()