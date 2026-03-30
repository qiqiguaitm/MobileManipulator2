import argparse
import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import requests
import yaml


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / 'src'
if str(SRC_DIR) not in sys.path:
	sys.path.append(str(SRC_DIR))

from pycocotools import mask as coco_mask


DATA_DIR = Path(__file__).resolve().parent / 'data'
OUTPUT_DIR = Path(__file__).resolve().parent / 'output'
PREDICT_ENDPOINT = '/api/predict'

OBJECT_COLORS_BGR = [
	(0, 0, 255),
	(0, 255, 0),
	(255, 0, 0),
	(0, 255, 255),
	(255, 0, 255),
	(255, 255, 0),
	(0, 128, 255),
	(128, 0, 255),
	(255, 128, 0),
	(128, 255, 0),
]


def parse_args():
	parser = argparse.ArgumentParser(description='Call grasp_server /api/predict and visualize results locally.')
	parser.add_argument('--server-url', default='http://14.103.44.161:17086', help='Base URL of deploy/grasp_server.py')
	parser.add_argument('--cdm-scene', default='scene1', help='Scene folder for CDM flow')
	parser.add_argument('--stereo-scene', default='scene2', help='Scene folder for stereo flow')
	parser.add_argument('--frame', default='000', help='Frame id without extension, e.g. 000')
	parser.add_argument('--text-prompt', default='object', help='Text prompt sent to predict API')
	parser.add_argument('--min-score', type=float, default=0.2, help='Minimum detection score')
	return parser.parse_args()


def load_intrinsics(scene_dir: Path):
	with scene_dir.joinpath('intrinsics.yaml').open('r', encoding='utf-8') as file_obj:
		data = yaml.safe_load(file_obj)

	color_profile = data['profiles'][0]
	ir_profile = data['ir_profiles'][0]
	return {
		'fx': float(color_profile['fx']),
		'fy': float(color_profile['fy']),
		'cx': float(color_profile['cx']),
		'cy': float(color_profile['cy']),
		'fx_ir': float(ir_profile['fx']),
		'baseline': float(data['baseline']),
	}


def scene_paths(scene_dir: Path, frame_id: str):
	return {
		'rgb': scene_dir / 'rgb' / f'{frame_id}.png',
		'depth': scene_dir / 'depth' / f'{frame_id}.png',
		'ir_left': scene_dir / 'ir_left_noproj' / f'{frame_id}.png',
		'ir_right': scene_dir / 'ir_right_noproj' / f'{frame_id}.png',
	}


def ensure_inputs_exist(paths):
	missing = [str(path) for path in paths.values() if not path.exists()]
	if missing:
		raise FileNotFoundError(f'Missing input files: {missing}')


def call_predict(server_url, mode, sample_paths, intrinsics, text_prompt, min_score):
	url = server_url.rstrip('/') + PREDICT_ENDPOINT
	data = {
		'depth_mode': mode,
		'text_prompt': text_prompt,
		'min_score': str(min_score),
	}
	if mode == 'stereo':
		data['fx_ir'] = str(intrinsics['fx_ir'])
		data['baseline'] = str(intrinsics['baseline'])

	with (
		sample_paths['rgb'].open('rb') as rgb_file,
		sample_paths['depth'].open('rb') as depth_file,
		sample_paths['ir_left'].open('rb') as ir_left_file,
		sample_paths['ir_right'].open('rb') as ir_right_file,
	):
		files = {
			'rgb_image': (sample_paths['rgb'].name, rgb_file, 'image/png'),
		}
		if mode == 'cdm':
			files['depth_image'] = (sample_paths['depth'].name, depth_file, 'image/png')
		elif mode == 'stereo':
			files['ir_left'] = (sample_paths['ir_left'].name, ir_left_file, 'image/png')
			files['ir_right'] = (sample_paths['ir_right'].name, ir_right_file, 'image/png')
		else:
			raise ValueError(f'Unsupported mode: {mode}')

		response = requests.post(url, data=data, files=files, timeout=300)

	response.raise_for_status()
	payload = response.json()
	if payload.get('error'):
		raise RuntimeError(payload['error'])
	return payload


def decode_mask_rle(mask_rle):
	if not mask_rle:
		return None
	counts = mask_rle.get('counts')
	if counts is None:
		return None

	if isinstance(counts, str):
		try:
			counts = base64.b64decode(counts)
		except Exception:
			counts = counts.encode('utf-8')

	rle = {
		'size': [int(v) for v in mask_rle['size']],
		'counts': counts,
	}
	return coco_mask.decode(rle).astype(bool)


def project_point(point_xyz, intrinsics):
	z = float(point_xyz[2])
	if z <= 1e-6:
		return None
	u = int(round(float(point_xyz[0]) * intrinsics['fx'] / z + intrinsics['cx']))
	v = int(round(float(point_xyz[1]) * intrinsics['fy'] / z + intrinsics['cy']))
	return u, v


def draw_grasp_2d(image, grasp, intrinsics, color, line_width=2):
	position = np.asarray(grasp['position'], dtype=np.float32)
	rotation = np.asarray(grasp['rotation'], dtype=np.float32)
	if rotation.shape != (3, 3):
		return

	finger_width = 0.08
	finger_length = 0.05
	tail_length = 0.03
	local_points = [
		np.array([finger_width / 2, 0, finger_length], dtype=np.float32),
		np.array([finger_width / 2, 0, 0], dtype=np.float32),
		np.array([-finger_width / 2, 0, 0], dtype=np.float32),
		np.array([-finger_width / 2, 0, finger_length], dtype=np.float32),
		np.array([0, 0, 0], dtype=np.float32),
		np.array([0, 0, -tail_length], dtype=np.float32),
	]
	world_points = [position + rotation @ point for point in local_points]
	points_2d = [project_point(point, intrinsics) for point in world_points]

	finger_pairs = [(0, 1), (1, 2), (2, 3)]
	for start_idx, end_idx in finger_pairs:
		start_pt, end_pt = points_2d[start_idx], points_2d[end_idx]
		if start_pt is not None and end_pt is not None:
			cv2.line(image, start_pt, end_pt, color, line_width, lineType=cv2.LINE_AA)

	center_pt, tail_pt = points_2d[4], points_2d[5]
	if center_pt is not None and tail_pt is not None:
		cv2.line(image, center_pt, tail_pt, color, line_width, lineType=cv2.LINE_AA)
		cv2.putText(
			image,
			f"{float(grasp.get('score', 0.0)):.2f}",
			(center_pt[0] + 5, center_pt[1] - 5),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.45,
			(255, 255, 255),
			1,
			cv2.LINE_AA,
		)


def visualize_2d(rgb_bgr, results, intrinsics, save_path: Path):
	vis = rgb_bgr.copy()
	category_counts = {}

	for obj_idx, obj in enumerate(results['objects']):
		color = OBJECT_COLORS_BGR[obj_idx % len(OBJECT_COLORS_BGR)]
		category = str(obj.get('category', 'object'))
		category_counts[category] = category_counts.get(category, 0) + 1
		label = f'{category} {category_counts[category]}'

		mask = decode_mask_rle(obj.get('mask_rle'))
		if mask is not None and mask.any():
			overlay = vis.copy()
			overlay[mask] = (0.65 * overlay[mask] + 0.35 * np.array(color, dtype=np.float32)).astype(np.uint8)
			vis = overlay
			contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			cv2.drawContours(vis, contours, -1, color, 2)

		bbox = obj.get('bbox')
		if bbox and len(bbox) >= 4:
			x1, y1, x2, y2 = [int(round(v)) for v in bbox[:4]]
			cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
			cv2.putText(
				vis,
				label,
				(x1, max(18, y1 - 8)),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.6,
				color,
				2,
				cv2.LINE_AA,
			)

		for grasp in obj.get('grasps', []):
			draw_grasp_2d(vis, grasp, intrinsics, color)

	save_path.parent.mkdir(parents=True, exist_ok=True)
	cv2.imwrite(str(save_path), vis)


def save_json(payload, save_path: Path):
	save_path.parent.mkdir(parents=True, exist_ok=True)
	with save_path.open('w', encoding='utf-8') as file_obj:
		json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def run_flow(mode, scene_name, frame_id, server_url, text_prompt, min_score):
	scene_dir = DATA_DIR / scene_name
	intrinsics = load_intrinsics(scene_dir)
	sample_paths = scene_paths(scene_dir, frame_id)
	ensure_inputs_exist(sample_paths)

	payload = call_predict(server_url, mode, sample_paths, intrinsics, text_prompt, min_score)
	results = payload['results']

	output_prefix = OUTPUT_DIR / f'{mode}_{scene_name}_{frame_id}'
	save_json(payload, output_prefix.with_suffix('.json'))

	rgb_bgr = cv2.imread(str(sample_paths['rgb']), cv2.IMREAD_COLOR)
	if rgb_bgr is None:
		raise ValueError(f'Failed to read RGB image: {sample_paths["rgb"]}')
	visualize_2d(rgb_bgr, results, intrinsics, output_prefix.with_name(output_prefix.name + '_2d.png'))

	print(f'[{mode}] objects={results["num_objects"]} saved to {output_prefix.parent}')


def main():
	args = parse_args()
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	run_flow(
		mode='cdm',
		scene_name=args.cdm_scene,
		frame_id=args.frame,
		server_url=args.server_url,
		text_prompt=args.text_prompt,
		min_score=args.min_score,
	)
	run_flow(
		mode='stereo',
		scene_name=args.stereo_scene,
		frame_id=args.frame,
		server_url=args.server_url,
		text_prompt=args.text_prompt,
		min_score=args.min_score,
	)


if __name__ == '__main__':
	main()
