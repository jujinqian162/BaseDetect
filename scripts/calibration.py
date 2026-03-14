"""Camera intrinsic calibration utility.

This script estimates intrinsic parameters from a set of calibration images
and can persist the results into ``configs/camera.yaml``.

Default workflow uses a classic chessboard because the project currently
depends on ``opencv-python`` instead of ``opencv-contrib-python``. If the
runtime OpenCV build exposes ``cv2.aruco``, the script can also calibrate from
ChArUco boards.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from basedetect.paths import camera_config


LOGGER = logging.getLogger("scripts.calibration")
ARUCO = getattr(cv2, "aruco", None)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class CalibrationResult:
	rms: float
	camera_matrix: np.ndarray
	dist_coeffs: np.ndarray
	image_width: int
	image_height: int
	per_view_errors: list[float]
	valid_images: list[Path]

	@property
	def fx(self) -> float:
		return float(self.camera_matrix[0, 0])

	@property
	def fy(self) -> float:
		return float(self.camera_matrix[1, 1])

	@property
	def cx(self) -> float:
		return float(self.camera_matrix[0, 2])

	@property
	def cy(self) -> float:
		return float(self.camera_matrix[1, 2])


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Estimate camera intrinsics from chessboard or ChArUco images."
	)
	parser.add_argument(
		"--input",
		type=Path,
		required=True,
		help="Image file, folder, or glob pattern containing calibration images.",
	)
	parser.add_argument(
		"--pattern",
		choices=("chessboard", "charuco"),
		default="chessboard",
		help="Calibration target type. Default uses standard chessboard corners.",
	)
	parser.add_argument(
		"--cols",
		type=int,
		default=9,
		help="Number of inner corners / squares in x direction depending on pattern.",
	)
	parser.add_argument(
		"--rows",
		type=int,
		default=6,
		help="Number of inner corners / squares in y direction depending on pattern.",
	)
	parser.add_argument(
		"--square-size",
		type=float,
		default=20.0,
		help="Physical square size in millimeters.",
	)
	parser.add_argument(
		"--marker-size",
		type=float,
		default=15.0,
		help="Marker size in millimeters for ChArUco boards.",
	)
	parser.add_argument(
		"--aruco-dict",
		default="DICT_4X4_50",
		help="Aruco dictionary name for ChArUco mode.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=camera_config(),
		help="YAML file to receive calibration results. Default: configs/camera.yaml",
	)
	parser.add_argument(
		"--save-debug",
		type=Path,
		default=None,
		help="Optional folder to save corner detection previews.",
	)
	parser.add_argument(
		"--show",
		action="store_true",
		help="Preview detections while processing images.",
	)
	return parser


def resolve_image_paths(input_path: Path) -> list[Path]:
	if any(char in str(input_path) for char in "*?[]"):
		matches = sorted(Path().glob(str(input_path)))
		return [path.resolve() for path in matches if path.suffix.lower() in IMAGE_EXTENSIONS]

	resolved = input_path.resolve()
	if resolved.is_file():
		return [resolved]
	if resolved.is_dir():
		return sorted(path.resolve() for path in resolved.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
	raise FileNotFoundError(f"Input path does not exist: {input_path}")


def build_chessboard_object_points(cols: int, rows: int, square_size: float) -> np.ndarray:
	object_points = np.zeros((rows * cols, 3), np.float32)
	grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
	object_points[:, :2] = grid * square_size
	return object_points


def _termination_criteria() -> tuple[int, int, float]:
	return (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)


def detect_chessboard_corners(
	image_path: Path,
	pattern_size: tuple[int, int],
	save_debug_dir: Path | None,
	show: bool,
) -> tuple[np.ndarray | None, tuple[int, int] | None]:
	image = cv2.imread(str(image_path))
	if image is None:
		LOGGER.warning("Skipping unreadable image: %s", image_path)
		return None, None

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
	found, corners = cv2.findChessboardCorners(gray, pattern_size, None, flags)
	if not found:
		LOGGER.warning("Chessboard corners not found in %s", image_path.name)
		return None, gray.shape[::-1]

	refined = cv2.cornerSubPix(
		gray,
		corners,
		winSize=(11, 11),
		zeroZone=(-1, -1),
		criteria=_termination_criteria(),
	)

	preview = image.copy()
	cv2.drawChessboardCorners(preview, pattern_size, refined, found)
	save_preview(preview, image_path, save_debug_dir)
	maybe_show_preview(preview, show)
	return refined, gray.shape[::-1]


def detect_charuco_corners(
	image_path: Path,
	cols: int,
	rows: int,
	square_size: float,
	marker_size: float,
	aruco_dict_name: str,
	save_debug_dir: Path | None,
	show: bool,
) -> tuple[np.ndarray | None, np.ndarray | None, tuple[int, int] | None]:
	if ARUCO is None:
		raise RuntimeError(
			"Current OpenCV build does not include cv2.aruco. Install opencv-contrib-python "
			"or use --pattern chessboard."
		)

	aruco_module = ARUCO
	dictionary_id = getattr(aruco_module, aruco_dict_name, None)
	if dictionary_id is None:
		raise ValueError(f"Unsupported ArUco dictionary: {aruco_dict_name}")

	dictionary = aruco_module.getPredefinedDictionary(dictionary_id)
	board = aruco_module.CharucoBoard((cols, rows), square_size, marker_size, dictionary)

	image = cv2.imread(str(image_path))
	if image is None:
		LOGGER.warning("Skipping unreadable image: %s", image_path)
		return None, None, None

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	detector_params = aruco_module.DetectorParameters()
	detector = aruco_module.ArucoDetector(dictionary, detector_params)
	marker_corners, marker_ids, _ = detector.detectMarkers(gray)
	if marker_ids is None or len(marker_ids) == 0:
		LOGGER.warning("No ArUco markers found in %s", image_path.name)
		return None, None, gray.shape[::-1]

	interpolate = getattr(aruco_module, "interpolateCornersCharuco", None)
	if interpolate is None:
		raise RuntimeError(
			"Current OpenCV build is missing interpolateCornersCharuco. "
			"Install opencv-contrib-python or use --pattern chessboard."
		)

	charuco_ret, charuco_corners, charuco_ids = interpolate(marker_corners, marker_ids, gray, board)
	if not charuco_ret:
		LOGGER.warning("ChArUco interpolation failed in %s", image_path.name)
		return None, None, gray.shape[::-1]
	if charuco_ids is None or len(charuco_ids) < 4:
		LOGGER.warning("Not enough ChArUco corners in %s", image_path.name)
		return None, None, gray.shape[::-1]

	preview = image.copy()
	aruco_module.drawDetectedMarkers(preview, marker_corners, marker_ids)
	aruco_module.drawDetectedCornersCharuco(preview, charuco_corners, charuco_ids)
	save_preview(preview, image_path, save_debug_dir)
	maybe_show_preview(preview, show)
	return charuco_corners, charuco_ids, gray.shape[::-1]


def save_preview(image: np.ndarray, image_path: Path, save_debug_dir: Path | None) -> None:
	if save_debug_dir is None:
		return
	save_debug_dir.mkdir(parents=True, exist_ok=True)
	output_path = save_debug_dir / f"{image_path.stem}_corners{image_path.suffix}"
	cv2.imwrite(str(output_path), image)


def maybe_show_preview(image: np.ndarray, show: bool) -> None:
	if not show:
		return
	cv2.imshow("calibration-preview", image)
	cv2.waitKey(250)


def calibrate_chessboard(
	image_paths: Iterable[Path],
	cols: int,
	rows: int,
	square_size: float,
	save_debug_dir: Path | None,
	show: bool,
) -> CalibrationResult:
	pattern_size = (cols, rows)
	template_object_points = build_chessboard_object_points(cols, rows, square_size)
	object_points: list[np.ndarray] = []
	image_points: list[np.ndarray] = []
	valid_images: list[Path] = []
	image_size: tuple[int, int] | None = None

	for image_path in image_paths:
		corners, current_size = detect_chessboard_corners(image_path, pattern_size, save_debug_dir, show)
		if current_size is not None:
			if image_size is None:
				image_size = current_size
			elif image_size != current_size:
				raise ValueError(
					f"All images must share the same resolution. {image_path.name} has {current_size}, expected {image_size}."
				)

		if corners is None:
			continue
		object_points.append(template_object_points.copy())
		image_points.append(corners)
		valid_images.append(image_path)

	if image_size is None or len(image_points) < 3:
		raise RuntimeError("Need at least 3 valid calibration images with detectable corners.")

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
	return CalibrationResult(
		rms=float(rms),
		camera_matrix=camera_matrix,
		dist_coeffs=dist_coeffs,
		image_width=image_size[0],
		image_height=image_size[1],
		per_view_errors=per_view_errors,
		valid_images=valid_images,
	)


def calibrate_charuco(
	image_paths: Iterable[Path],
	cols: int,
	rows: int,
	square_size: float,
	marker_size: float,
	aruco_dict_name: str,
	save_debug_dir: Path | None,
	show: bool,
) -> CalibrationResult:
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

	all_charuco_corners: list[np.ndarray] = []
	all_charuco_ids: list[np.ndarray] = []
	valid_images: list[Path] = []
	image_size: tuple[int, int] | None = None

	for image_path in image_paths:
		corners, ids, current_size = detect_charuco_corners(
			image_path,
			cols,
			rows,
			square_size,
			marker_size,
			aruco_dict_name,
			save_debug_dir,
			show,
		)
		if current_size is not None:
			if image_size is None:
				image_size = current_size
			elif image_size != current_size:
				raise ValueError(
					f"All images must share the same resolution. {image_path.name} has {current_size}, expected {image_size}."
				)

		if corners is None or ids is None:
			continue
		all_charuco_corners.append(corners)
		all_charuco_ids.append(ids)
		valid_images.append(image_path)

	if image_size is None or len(all_charuco_corners) < 3:
		raise RuntimeError("Need at least 3 valid ChArUco images with enough detected corners.")

	calibrate_charuco = getattr(aruco_module, "calibrateCameraCharuco", None)
	if calibrate_charuco is None:
		raise RuntimeError(
			"Current OpenCV build is missing calibrateCameraCharuco. Install opencv-contrib-python or use --pattern chessboard."
		)

	initial_camera_matrix = np.eye(3, dtype=np.float64)
	initial_dist_coeffs = np.zeros((8, 1), dtype=np.float64)
	rms, camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_charuco(
		all_charuco_corners,
		all_charuco_ids,
		board,
		image_size,
		initial_camera_matrix,
		initial_dist_coeffs,
	)

	object_points: list[np.ndarray] = []
	image_points: list[np.ndarray] = []
	for corners, ids in zip(all_charuco_corners, all_charuco_ids):
		chessboard_corners = np.asarray(board.getChessboardCorners(), dtype=np.float32)[
			ids.flatten().tolist()
		]
		object_points.append(chessboard_corners.astype(np.float32))
		image_points.append(corners.astype(np.float32))

	per_view_errors = compute_reprojection_errors(
		object_points, image_points, rvecs, tvecs, camera_matrix, dist_coeffs
	)
	return CalibrationResult(
		rms=float(rms),
		camera_matrix=camera_matrix,
		dist_coeffs=dist_coeffs,
		image_width=image_size[0],
		image_height=image_size[1],
		per_view_errors=per_view_errors,
		valid_images=valid_images,
	)


def compute_reprojection_errors(
	object_points: list[np.ndarray],
	image_points: list[np.ndarray],
	rvecs: Sequence[np.ndarray],
	tvecs: Sequence[np.ndarray],
	camera_matrix: np.ndarray,
	dist_coeffs: np.ndarray,
) -> list[float]:
	errors: list[float] = []
	for obj_pts, img_pts, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
		projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)
		error = cv2.norm(img_pts, projected, cv2.NORM_L2) / len(projected)
		errors.append(float(error))
	return errors


def update_camera_yaml(output_path: Path, result: CalibrationResult) -> None:
	existing: dict = {}
	if output_path.exists():
		with output_path.open("r", encoding="utf-8") as file:
			loaded = yaml.safe_load(file) or {}
			if isinstance(loaded, dict):
				existing = loaded

	camera_section = existing.get("camera", {})
	if not isinstance(camera_section, dict):
		camera_section = {}

	camera_section.update(
		{
			"image_width": int(result.image_width),
			"image_height": int(result.image_height),
			"fx": round(result.fx, 6),
			"fy": round(result.fy, 6),
			"cx": round(result.cx, 6),
			"cy": round(result.cy, 6),
			"dist_coeffs": [round(float(value), 8) for value in result.dist_coeffs.reshape(-1)],
			"rms_reprojection_error": round(result.rms, 8),
			"mean_reprojection_error": round(float(np.mean(result.per_view_errors)), 8),
		}
	)
	existing["camera"] = camera_section

	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as file:
		yaml.safe_dump(existing, file, allow_unicode=True, sort_keys=False)


def print_summary(result: CalibrationResult, output_path: Path) -> None:
	mean_error = float(np.mean(result.per_view_errors)) if result.per_view_errors else float("nan")
	print("Calibration succeeded")
	print(f"  images used: {len(result.valid_images)}")
	print(f"  image size : {result.image_width} x {result.image_height}")
	print(f"  fx, fy     : {result.fx:.6f}, {result.fy:.6f}")
	print(f"  cx, cy     : {result.cx:.6f}, {result.cy:.6f}")
	print(f"  distortion : {result.dist_coeffs.reshape(-1).tolist()}")
	print(f"  rms error  : {result.rms:.6f}")
	print(f"  mean error : {mean_error:.6f}")
	print(f"  saved to   : {output_path}")


def main() -> None:
	logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
	args = build_parser().parse_args()

	if args.square_size <= 0:
		raise ValueError("--square-size must be positive.")
	if args.pattern == "charuco" and args.marker_size <= 0:
		raise ValueError("--marker-size must be positive.")

	image_paths = resolve_image_paths(args.input)
	if len(image_paths) < 3:
		raise RuntimeError("Please provide at least 3 calibration images.")

	LOGGER.info("Found %d candidate images.", len(image_paths))
	try:
		if args.pattern == "chessboard":
			result = calibrate_chessboard(
				image_paths,
				cols=args.cols,
				rows=args.rows,
				square_size=args.square_size,
				save_debug_dir=args.save_debug,
				show=args.show,
			)
		else:
			result = calibrate_charuco(
				image_paths,
				cols=args.cols,
				rows=args.rows,
				square_size=args.square_size,
				marker_size=args.marker_size,
				aruco_dict_name=args.aruco_dict,
				save_debug_dir=args.save_debug,
				show=args.show,
			)
	finally:
		if args.show:
			cv2.destroyAllWindows()

	update_camera_yaml(args.output, result)
	print_summary(result, args.output)


if __name__ == "__main__":
	main()
