"""Save failure evidence without moving the robot; optional synchronized capture happens after stopping."""
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np


def save_failure(directory, stage, error, node, camera_transform=None):
    destination = Path(directory) / 'failure'
    destination.mkdir(exist_ok=True)
    report = {'error': str(error), 'stage': str(stage) if stage else None,
              'images': [], 'save_errors': [], 'camera_image': None}
    if stage is not None and (Path(stage)/'async').exists():
        report['async_evidence_dir'] = str(Path(stage)/'async')
    # Keep every frame: a consistency failure belongs to the entire observation set.
    if stage is not None:
        for source in sorted(Path(stage).iterdir()):
            if source.is_file() and source.suffix in ('.png', '.npy', '.npz', '.json', '.pcd'):
                try:
                    shutil.copy2(source, destination / source.name)
                    if source.suffix == '.png':
                        report['images'].append(source.name)
                except Exception as exc:
                    report['save_errors'].append(f'{source.name}: {exc}')
    # This is the last received frame, possibly stale, not a new synchronized observation.
    message = node.data.get('rgb')
    if message is not None:
        try:
            if message.encoding not in ('rgb8', 'bgr8'):
                raise ValueError('Unsupported RGB encoding: ' + message.encoding)
            pixels = np.frombuffer(message.data, np.uint8).reshape(message.height, message.step)
            pixels = pixels[:, :message.width * 3].reshape(message.height, message.width, 3).copy()
            if message.encoding == 'rgb8':
                pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(destination / 'last_received_camera.png'), pixels):
                raise OSError('Could not save camera image')
            report['camera_image'] = {
                'file': 'last_received_camera.png', 'source': 'last_received_not_new_capture',
                'age_seconds': time.monotonic() - node.received.get('rgb', 0),
                'timestamp_ns': message.header.stamp.sec * 10**9 + message.header.stamp.nanosec}
        except Exception as exc:
            report['save_errors'].append(str(exc))
    if 'Obstacle in ' in str(error) and camera_transform is not None:
        try:
            from .obstacle_evidence import save_after_stop
            report['after_stop_capture']=save_after_stop(destination,node,camera_transform(node))
            report['images'].extend(['after_stop_camera.png','after_stop_lidar_overlay.png','after_stop_lidar_top.png'])
        except Exception as exc:
            # Trigger scan and last-received camera remain even if RGB/3D sync fails.
            report['save_errors'].append('after_stop_capture: '+str(exc))
    (destination / 'failure.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return str(destination)
