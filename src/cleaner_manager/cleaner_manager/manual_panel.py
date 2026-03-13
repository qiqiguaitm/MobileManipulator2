"""
manual_panel.py — PyQt5 standalone GUI for manual pick-and-place control.

Subscribes to ManualStatus, TracerStatus, PiperStatus.
Sends ManualCommand service calls to ManualControlNode.
"""

import sys
import signal

import rclpy
from rclpy.node import Node

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QListWidget, QListWidgetItem,
    QProgressBar, QSizePolicy,
)
from PyQt5.QtCore import Qt, QTimer, QMetaObject, pyqtSlot, Q_ARG
from PyQt5.QtGui import QFont

from cleaner_manager.msg import ManualStatus
from cleaner_manager.srv import ManualCommand
from tracer_msgs.msg import TracerStatus
from piper_msgs.msg import PiperStatus
from perception.msg import PerceptionConfig
from std_msgs.msg import String
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy


# State constants
STATE_IDLE = 0
STATE_DETECTING = 1
STATE_WAITING_SELECT = 2
STATE_WAITING_CONFIRM = 3
STATE_NAVIGATING = 4
STATE_APPROACHING = 5
STATE_OBSERVING = 6
STATE_PICKING = 7
STATE_PLACING = 8
STATE_ERROR = 9

_STATE_LABELS = {
    STATE_IDLE: "空闲",
    STATE_DETECTING: "检测中...",
    STATE_WAITING_SELECT: "请选择目标",
    STATE_WAITING_CONFIRM: "请确认执行",
    STATE_NAVIGATING: "导航中...",
    STATE_APPROACHING: "接近中...",
    STATE_OBSERVING: "观察中...",
    STATE_PICKING: "抓取中...",
    STATE_PLACING: "放置中...",
    STATE_ERROR: "错误",
}

_STATE_COLORS = {
    STATE_IDLE: "#888888",
    STATE_DETECTING: "#2196F3",
    STATE_WAITING_SELECT: "#FF9800",
    STATE_WAITING_CONFIRM: "#FF9800",
    STATE_NAVIGATING: "#2196F3",
    STATE_APPROACHING: "#2196F3",
    STATE_OBSERVING: "#9C27B0",
    STATE_PICKING: "#4CAF50",
    STATE_PLACING: "#4CAF50",
    STATE_ERROR: "#F44336",
}


class ManualPanel(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("手动控制面板")
        self.setMinimumWidth(420)

        # ROS2 node
        self._node = rclpy.create_node('manual_panel')

        # Subscriptions
        self._node.create_subscription(
            ManualStatus, '/manual_control_node/status',
            self._status_cb, 10)
        self._node.create_subscription(
            TracerStatus, '/tracer_status',
            self._chassis_cb, 10)
        self._node.create_subscription(
            PiperStatus, '/piper/status',
            self._arm_status_cb, 10)

        # Subscribe to default_prompt from perception (single source of truth, latched)
        _latched_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE)
        self._node.create_subscription(
            String, '/piper/default_prompt',
            self._default_prompt_cb, _latched_qos)

        # Service client
        self._cmd_cli = self._node.create_client(
            ManualCommand, '/manual_control_node/command')

        # Publisher for auto-detection config
        self._config_pub = self._node.create_publisher(
            PerceptionConfig, '/perception/config', 1)

        # Latest messages
        self._last_status = None
        self._last_chassis = None
        self._last_arm = None

        self._build_ui()

        # QTimer drives ROS spin (20Hz)
        self._ros_timer = QTimer()
        self._ros_timer.timeout.connect(self._ros_spin)
        self._ros_timer.start(50)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        # --- State ---
        grp = QGroupBox("状态")
        h = QHBoxLayout(grp)
        self._state_dot = QLabel("●")
        self._state_dot.setFont(QFont("", 14))
        self._state_label = QLabel("空闲")
        self._state_label.setFont(QFont("", 12, QFont.Bold))
        self._state_msg = QLabel("")
        self._state_msg.setWordWrap(True)
        h.addWidget(self._state_dot)
        h.addWidget(self._state_label)
        h.addWidget(self._state_msg, 1)
        layout.addWidget(grp)

        # --- Chassis ---
        grp = QGroupBox("底盘")
        v = QVBoxLayout(grp)
        self._chassis_label = QLabel("● 未连接")
        v.addWidget(self._chassis_label)
        self._battery_bar = QProgressBar()
        self._battery_bar.setMaximum(100)
        self._battery_bar.setTextVisible(True)
        self._battery_bar.setMaximumHeight(16)
        v.addWidget(self._battery_bar)
        layout.addWidget(grp)

        # --- Arm ---
        grp = QGroupBox("机械臂")
        h = QHBoxLayout(grp)
        self._arm_label = QLabel("● 未连接")
        h.addWidget(self._arm_label)
        layout.addWidget(grp)

        # --- Detection ---
        grp = QGroupBox("自动检测")
        v = QVBoxLayout(grp)
        h = QHBoxLayout()
        self._prompt_edit = QLineEdit("bottle.cup.box.can.toy.pen.bag")
        self._prompt_edit.setPlaceholderText("检测提示词")
        self._prompt_edit.returnPressed.connect(self._on_set_prompt)
        h.addWidget(self._prompt_edit, 1)
        self._btn_set_prompt = QPushButton("应用")
        self._btn_set_prompt.clicked.connect(self._on_set_prompt)
        h.addWidget(self._btn_set_prompt)
        v.addLayout(h)
        layout.addWidget(grp)

        # --- Candidates ---
        grp = QGroupBox("候选目标")
        v = QVBoxLayout(grp)
        self._candidate_list = QListWidget()
        self._candidate_list.setMaximumHeight(120)
        self._candidate_list.currentRowChanged.connect(self._on_candidate_select)
        v.addWidget(self._candidate_list)
        layout.addWidget(grp)

        # --- Target info ---
        grp = QGroupBox("目标信息")
        v = QVBoxLayout(grp)
        self._target_label = QLabel("未选择")
        self._target_label.setWordWrap(True)
        v.addWidget(self._target_label)
        layout.addWidget(grp)

        # --- Grasp progress ---
        self._progress_grp = QGroupBox("进度")
        v = QVBoxLayout(self._progress_grp)
        h2 = QHBoxLayout()
        self._progress_bar = QProgressBar()
        self._progress_bar.setMaximum(100)
        h2.addWidget(self._progress_bar, 1)
        self._progress_step = QLabel("")
        h2.addWidget(self._progress_step)
        v.addLayout(h2)
        self._progress_grp.setVisible(False)
        layout.addWidget(self._progress_grp)

        # --- Action buttons ---
        h = QHBoxLayout()
        self._btn_go = QPushButton("GO")
        self._btn_go.setMinimumHeight(48)
        self._btn_go.setFont(QFont("", 14, QFont.Bold))
        self._btn_go.setStyleSheet(
            "QPushButton { background: #4CAF50; color: white; border-radius: 6px; }"
            "QPushButton:disabled { background: #ccc; color: #888; }"
            "QPushButton:hover:!disabled { background: #388E3C; }")
        self._btn_go.clicked.connect(self._on_go)
        h.addWidget(self._btn_go)

        self._btn_cancel = QPushButton("CANCEL")
        self._btn_cancel.setMinimumHeight(48)
        self._btn_cancel.setFont(QFont("", 14, QFont.Bold))
        self._btn_cancel.setStyleSheet(
            "QPushButton { background: #F44336; color: white; border-radius: 6px; }"
            "QPushButton:disabled { background: #ccc; color: #888; }"
            "QPushButton:hover:!disabled { background: #C62828; }")
        self._btn_cancel.clicked.connect(self._on_cancel)
        h.addWidget(self._btn_cancel)
        layout.addLayout(h)

        # Initial button states
        self._update_buttons(STATE_IDLE)

    # ------------------------------------------------------------------
    # ROS spin
    # ------------------------------------------------------------------

    def _ros_spin(self):
        rclpy.spin_once(self._node, timeout_sec=0)

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------

    def _status_cb(self, msg):
        self._last_status = msg
        # Schedule UI update on Qt main thread
        QMetaObject.invokeMethod(
            self, '_update_status_ui', Qt.QueuedConnection)

    def _chassis_cb(self, msg):
        self._last_chassis = msg
        QMetaObject.invokeMethod(
            self, '_update_chassis_ui', Qt.QueuedConnection)

    def _arm_status_cb(self, msg):
        self._last_arm = msg
        QMetaObject.invokeMethod(
            self, '_update_arm_ui', Qt.QueuedConnection)

    def _default_prompt_cb(self, msg):
        if msg.data:
            QMetaObject.invokeMethod(
                self._prompt_edit, 'setText', Qt.QueuedConnection,
                Q_ARG(str, msg.data))

    # ------------------------------------------------------------------
    # UI updates (Qt main thread)
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _update_status_ui(self):
        msg = self._last_status
        if msg is None:
            return

        state = msg.state
        color = _STATE_COLORS.get(state, "#888")
        label = _STATE_LABELS.get(state, "未知")

        self._state_dot.setStyleSheet(f"color: {color};")
        self._state_label.setText(label)
        self._state_label.setStyleSheet(f"color: {color};")
        self._state_msg.setText(msg.message)

        # Update prompt from backend
        if msg.current_prompt and msg.current_prompt != self._prompt_edit.text():
            self._prompt_edit.setText(msg.current_prompt)

        # Update candidate list
        if state in (STATE_WAITING_SELECT, STATE_WAITING_CONFIRM):
            self._update_candidate_list(msg.candidates, msg.selected_index)
        elif state == STATE_IDLE:
            if not msg.candidates:
                self._candidate_list.clear()

        # Update target info
        if state >= STATE_WAITING_CONFIRM and msg.target_category:
            self._target_label.setText(
                f"类别: {msg.target_category} ({msg.target_object_id})\n"
                f"距离: {msg.target_distance:.2f} m\n"
                f"目标: ({msg.target_position.x:.2f}, {msg.target_position.y:.2f})  "
                f"底盘: ({msg.robot_position.x:.2f}, {msg.robot_position.y:.2f})")
        elif state == STATE_IDLE:
            self._target_label.setText("未选择")

        # Grasp progress
        show_progress = state in (STATE_PICKING, STATE_PLACING)
        self._progress_grp.setVisible(show_progress)
        if show_progress:
            self._progress_bar.setValue(int(msg.grasp_progress * 100))
            self._progress_step.setText(msg.grasp_step)

        self._update_buttons(state)

    @pyqtSlot()
    def _update_chassis_ui(self):
        msg = self._last_chassis
        if msg is None:
            return
        voltage = msg.battery_voltage
        pct = max(0, min(100, (voltage - 24.0) / (29.4 - 24.0) * 100))
        self._battery_bar.setValue(int(pct))
        self._battery_bar.setFormat(f"{pct:.0f}%  {voltage:.1f}V")
        if pct > 50:
            chunk_color = '#4CAF50'
        elif pct > 20:
            chunk_color = '#FF9800'
        else:
            chunk_color = '#F44336'
        self._battery_bar.setStyleSheet(
            f"QProgressBar::chunk {{ background: {chunk_color}; }}")

        health = '● 健康' if msg.error_code == 0 else f'● 故障 {msg.error_code}'
        self._chassis_label.setText(
            f"{health}  {msg.linear_velocity:.2f} m/s  {msg.angular_velocity:.2f} rad/s")

    @pyqtSlot()
    def _update_arm_ui(self):
        msg = self._last_arm
        if msg is None:
            return
        if not msg.connected:
            self._arm_label.setText("● 未连接")
            self._arm_label.setStyleSheet("color: #F44336;")
        elif not msg.enabled:
            self._arm_label.setText("● 已连接 (未使能)")
            self._arm_label.setStyleSheet("color: #FF9800;")
        elif msg.error_state != "NORMAL":
            self._arm_label.setText(f"● {msg.error_state}: {msg.error_description}")
            self._arm_label.setStyleSheet("color: #F44336;")
        else:
            self._arm_label.setText("● 已使能  正常")
            self._arm_label.setStyleSheet("color: #4CAF50;")

    def _update_candidate_list(self, candidates, selected_index):
        # Only rebuild if content changed
        if self._candidate_list.count() != len(candidates):
            self._candidate_list.blockSignals(True)
            self._candidate_list.clear()
            for c in candidates:
                text = f"{c.category} ({c.object_id}) - {c.distance:.2f}m  [{c.score:.0%}]"
                self._candidate_list.addItem(text)
            self._candidate_list.blockSignals(False)

        if 0 <= selected_index < self._candidate_list.count():
            self._candidate_list.blockSignals(True)
            self._candidate_list.setCurrentRow(selected_index)
            self._candidate_list.blockSignals(False)

    def _update_buttons(self, state):
        idle_or_select = state in (STATE_IDLE, STATE_WAITING_SELECT)
        busy = state in (STATE_DETECTING, STATE_NAVIGATING, STATE_APPROACHING,
                         STATE_OBSERVING, STATE_PICKING, STATE_PLACING)

        self._btn_set_prompt.setEnabled(idle_or_select)
        self._btn_go.setEnabled(state == STATE_WAITING_CONFIRM)
        self._btn_cancel.setEnabled(busy or state == STATE_WAITING_SELECT
                                    or state == STATE_WAITING_CONFIRM)
        self._candidate_list.setEnabled(state == STATE_WAITING_SELECT)

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_set_prompt(self):
        prompt = self._prompt_edit.text().strip()
        if not prompt:
            return
        # 同步更新 manual_control_node 的 prompt（用于抓取流程）
        self._send_command(ManualCommand.Request.CMD_SET_PROMPT, prompt=prompt)
        # 直接更新自动检测 prompt
        msg = PerceptionConfig()
        msg.prompt = prompt
        msg.min_score = 0.0  # 0 表示不修改
        self._config_pub.publish(msg)

    def _on_candidate_select(self, row):
        if row >= 0:
            self._send_command(ManualCommand.Request.CMD_SELECT, target_index=row)

    def _on_go(self):
        self._send_command(ManualCommand.Request.CMD_GO)

    def _on_cancel(self):
        self._send_command(ManualCommand.Request.CMD_CANCEL)

    def _send_command(self, cmd, prompt="", target_index=0):
        req = ManualCommand.Request()
        req.command = cmd
        req.prompt = prompt
        req.target_index = target_index
        self._cmd_cli.call_async(req)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        self._ros_timer.stop()
        self._node.destroy_node()
        super().closeEvent(event)


def main():
    rclpy.init()
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication(sys.argv)
    panel = ManualPanel()
    panel.show()

    ret = app.exec_()
    rclpy.shutdown()
    sys.exit(ret)


if __name__ == '__main__':
    main()
