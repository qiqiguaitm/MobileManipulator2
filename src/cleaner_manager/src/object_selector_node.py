#!/usr/bin/env python3
"""
object_selector_node.py — Interactive 3D object selector for RViz.

Subscribes to fused/objects_3d (always), renders clickable InteractiveMarkers.
Clicking a marker sends CMD_SELECT_OBJECT to manual_control_node.

Perception guarantees objects are in 'map' frame (source gate).
No TF lookup needed — coordinates are used directly.
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, Marker
from interactive_markers import InteractiveMarkerServer

from perception.msg import Object3DArray
from cleaner_manager.srv import ManualCommand

# Color palette for objects (R, G, B)
_COLORS = [
    (0.2, 0.6, 1.0),
    (0.2, 0.8, 0.4),
    (1.0, 0.4, 0.2),
    (0.8, 0.2, 0.8),
    (0.0, 0.8, 0.8),
    (1.0, 0.6, 0.0),
]


class ObjectSelectorNode(Node):

    def __init__(self):
        super().__init__('object_selector_node')
        self._log = self.get_logger()
        self._cb_group = ReentrantCallbackGroup()

        self._server = InteractiveMarkerServer(self, 'object_selector')

        self._cmd_cli = self.create_client(
            ManualCommand, '/manual_control_node/command',
            callback_group=self._cb_group)

        self.create_subscription(
            Object3DArray,
            '/multi_camera_perception/fused/objects_3d',
            self._objects_cb, 10,
            callback_group=self._cb_group)

        # Cached objects for timer-driven marker rebuild
        self._cached_objects = []
        self._cached_dirty = False
        self._prev_marker_ids = set()

        # Single timer drives all marker updates (1Hz) to avoid flooding
        # the InteractiveMarkerServer and causing sequence-number warnings.
        self.create_timer(1.0, self._update_timer_cb,
                          callback_group=self._cb_group)

        self._log.info('ObjectSelectorNode initialized (always-on mode)')

    def _objects_cb(self, msg: Object3DArray):
        """Cache latest fused detections; timer will rebuild markers."""
        if len(msg.objects) == 0:
            if self._cached_objects:
                self._cached_objects = []
                self._cached_dirty = True
            return

        self._cached_objects = msg.objects
        self._cached_dirty = True

    def _update_timer_cb(self):
        """Rebuild markers at fixed 1Hz from cached data."""
        if not self._cached_dirty and not self._cached_objects:
            return

        if not self._cached_objects:
            # Clear all markers
            if self._prev_marker_ids:
                for mid in self._prev_marker_ids:
                    self._server.erase(mid)
                self._server.applyChanges()
                self._prev_marker_ids.clear()
            self._cached_dirty = False
            return

        self._build_markers(self._cached_objects)
        self._cached_dirty = False

    def _build_markers(self, objects):
        new_ids = {obj.object_id for obj in objects}

        # Remove markers for objects no longer present
        for old_id in self._prev_marker_ids - new_ids:
            self._server.erase(old_id)

        # Insert/update markers for current objects (already in map frame)
        for idx, obj in enumerate(objects):
            pos = obj.position
            name = obj.object_id
            r, g, b = _COLORS[idx % len(_COLORS)]

            im = InteractiveMarker()
            im.header.frame_id = 'map'
            im.name = name
            im.description = ''
            im.pose.position.x = pos.x
            im.pose.position.y = pos.y
            im.pose.position.z = pos.z
            im.scale = 0.2

            ctrl = InteractiveMarkerControl()
            ctrl.interaction_mode = InteractiveMarkerControl.BUTTON
            ctrl.always_visible = True

            cube = Marker()
            cube.type = Marker.CUBE
            cube.scale.x = 0.15
            cube.scale.y = 0.15
            cube.scale.z = 0.15
            cube.color.r = float(r)
            cube.color.g = float(g)
            cube.color.b = float(b)
            cube.color.a = 0.75
            ctrl.markers.append(cube)

            text = Marker()
            text.type = Marker.TEXT_VIEW_FACING
            text.text = f'{obj.category}\n{obj.distance:.1f}m {obj.score:.2f}'
            text.scale.z = 0.07
            text.pose.position.z = 0.15
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0
            ctrl.markers.append(text)

            im.controls.append(ctrl)
            self._server.insert(
                im,
                feedback_callback=self._make_click_cb(obj, pos))

        self._server.applyChanges()
        self._prev_marker_ids = new_ids
        self._log.debug(
            f'markers applied: {len(new_ids)} objects in map frame')

    def _make_click_cb(self, obj, pos):
        def cb(feedback):
            if feedback.event_type != feedback.BUTTON_CLICK:
                return
            self._log.info(f'Marker clicked: {obj.object_id} ({obj.category})')
            req = ManualCommand.Request()
            req.command = ManualCommand.Request.CMD_SELECT_OBJECT
            req.object_id = obj.object_id
            req.category = obj.category
            req.distance = obj.distance
            req.target_position.x = pos.x
            req.target_position.y = pos.y
            req.target_position.z = pos.z
            req.position_frame = 'map'
            self._cmd_cli.call_async(req)
        return cb


def main(args=None):
    rclpy.init(args=args)
    node = ObjectSelectorNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
