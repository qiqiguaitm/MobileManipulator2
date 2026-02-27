/**
 * @file piper_grasp_panel.cpp
 * @brief RViz2 Panel for Piper Grasp Control - ROS2 Implementation
 *
 * Units:
 *   - Position: mm (X, Y, Z)
 *   - Angles: degrees (Roll, Pitch, Yaw, Joint angles)
 *   - Gripper: mm (0 = closed, 70 = fully open)
 */

#include "piper_grasp/piper_grasp_panel.h"
#include <pluginlib/class_list_macros.hpp>
#include <QApplication>
#include <QDateTime>
#include <thread>
#include <chrono>
#include <cmath>

namespace piper_grasp
{

// Define static constexpr members
constexpr double PiperGraspPanel::GRIPPER_MAX_MM;
constexpr int PiperGraspPanel::ERROR_DISPLAY_DURATION_MS;
constexpr int PiperGraspPanel::CAMERA_SETTLE_TIME_MS;
constexpr int PiperGraspPanel::AUTO_CYCLE_DELAY_SUCCESS_MS;
constexpr int PiperGraspPanel::AUTO_CYCLE_DELAY_FAILURE_MS;
constexpr int PiperGraspPanel::AUTO_MAX_CONSECUTIVE_FAILURES;
constexpr int PiperGraspPanel::AUTO_MAX_CONSECUTIVE_EMPTY;

// Helper: Convert std::vector<float> to QVector<double>
static QVector<double> toQVector(const std::vector<float>& v)
{
    QVector<double> result;
    for (float f : v) result.append(static_cast<double>(f));
    return result;
}

PiperGraspPanel::PiperGraspPanel(QWidget* parent)
    : rviz_common::Panel(parent)
{
    // Initialize timers
    error_clear_timer_ = new QTimer(this);
    error_clear_timer_->setSingleShot(true);
    connect(error_clear_timer_, &QTimer::timeout, this, [this]() {
        if (!panel_alive_.load()) return;
        label_error_display_->setVisible(false);
        label_error_display_->setText("");
    });

    auto_cycle_timer_ = new QTimer(this);
    auto_cycle_timer_->setSingleShot(true);
    connect(auto_cycle_timer_, &QTimer::timeout, this, &PiperGraspPanel::startNextAutoCycle);

    // Spin timer for ROS2
    spin_timer_ = new QTimer(this);
    connect(spin_timer_, &QTimer::timeout, this, &PiperGraspPanel::spinOnce);

    setupUi();

    // Connect signals (queued for thread safety)
    connect(this, &PiperGraspPanel::statusUpdated,
            this, &PiperGraspPanel::handleStatusUpdated, Qt::QueuedConnection);
    connect(this, &PiperGraspPanel::armStatusUpdated,
            this, &PiperGraspPanel::handleArmStatusUpdated, Qt::QueuedConnection);
    connect(this, &PiperGraspPanel::serviceResult,
            this, &PiperGraspPanel::handleServiceResult, Qt::QueuedConnection);
    connect(this, &PiperGraspPanel::observeResult,
            this, &PiperGraspPanel::handleObserveResult, Qt::QueuedConnection);
    connect(this, &PiperGraspPanel::pickResult,
            this, &PiperGraspPanel::handlePickResult, Qt::QueuedConnection);
    connect(this, &PiperGraspPanel::placeResult,
            this, &PiperGraspPanel::handlePlaceResult, Qt::QueuedConnection);
    connect(this, &PiperGraspPanel::pickProgress,
            this, &PiperGraspPanel::handlePickProgress, Qt::QueuedConnection);
    connect(this, &PiperGraspPanel::placeProgress,
            this, &PiperGraspPanel::handlePlaceProgress, Qt::QueuedConnection);

    // Default prompt
    current_prompt_ = "bottle.cup.toy.box.pen.key";
    edit_prompt_->setText(current_prompt_);
}

void PiperGraspPanel::onInitialize()
{
    // Get display context with null check
    auto context = getDisplayContext();
    if (!context) {
        RCLCPP_ERROR(rclcpp::get_logger("piper_grasp_panel"), "Display context is null");
        return;
    }

    // Get node from RViz context
    auto ros_node_abstraction = context->getRosNodeAbstraction().lock();
    if (!ros_node_abstraction) {
        RCLCPP_ERROR(rclcpp::get_logger("piper_grasp_panel"), "Failed to get ROS node abstraction");
        return;
    }

    // Create our own node for subscriptions and services
    // Use unique node name to avoid conflicts
    try {
        auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        std::string node_name = "piper_grasp_panel_" + std::to_string(now % 100000);
        node_ = std::make_shared<rclcpp::Node>(node_name);
        executor_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
        executor_->add_node(node_);

        setupRos();

        // Start spin timer (10Hz)
        spin_timer_->start(100);

        RCLCPP_INFO(node_->get_logger(), "[PiperGraspPanel] ROS2 panel initialized");
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("piper_grasp_panel"), "Failed to initialize: %s", e.what());
    }
}

PiperGraspPanel::~PiperGraspPanel()
{
    panel_alive_.store(false);

    if (spin_timer_) {
        spin_timer_->stop();
    }
    if (error_clear_timer_) {
        error_clear_timer_->stop();
    }
    if (auto_cycle_timer_) {
        auto_cycle_timer_->stop();
    }

    // Cancel any running actions (with null checks)
    if (pick_client_ && current_pick_goal_) {
        try {
            pick_client_->async_cancel_goal(current_pick_goal_);
        } catch (...) {}
    }
    if (place_client_ && current_place_goal_) {
        try {
            place_client_->async_cancel_goal(current_place_goal_);
        } catch (...) {}
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    if (executor_) {
        try {
            executor_->cancel();
        } catch (...) {}
    }
}

void PiperGraspPanel::spinOnce()
{
    if (executor_) {
        executor_->spin_some(std::chrono::milliseconds(10));
    }
}

void PiperGraspPanel::setupUi()
{
    // Use QScrollArea to make panel scrollable in RViz2
    QVBoxLayout* outer_layout = new QVBoxLayout(this);
    outer_layout->setSpacing(0);
    outer_layout->setContentsMargins(0, 0, 0, 0);

    QScrollArea* scroll_area = new QScrollArea();
    scroll_area->setWidgetResizable(true);
    scroll_area->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    scroll_area->setFrameShape(QFrame::NoFrame);

    QWidget* scroll_content = new QWidget();
    QVBoxLayout* main_layout = new QVBoxLayout(scroll_content);
    main_layout->setSpacing(5);
    main_layout->setContentsMargins(5, 5, 5, 5);

    // === Title ===
    QLabel* title = new QLabel("<b>PIPER GRASP CONTROL</b>");
    title->setAlignment(Qt::AlignCenter);
    main_layout->addWidget(title);

    // === Top Bar: Connection + Error + STOP ===
    QHBoxLayout* top_layout = new QHBoxLayout();
    label_connection_ = new QLabel("DISCONNECTED");
    label_connection_->setStyleSheet("color: red; font-weight: bold;");
    top_layout->addWidget(label_connection_);
    top_layout->addStretch();
    label_error_ = new QLabel("Error: --");
    top_layout->addWidget(label_error_);
    top_layout->addStretch();
    btn_stop_ = new QPushButton("STOP");
    btn_stop_->setStyleSheet("background-color: #FF0000; color: white; font-weight: bold;");
    btn_stop_->setMinimumWidth(60);
    connect(btn_stop_, &QPushButton::clicked, this, &PiperGraspPanel::onStopClicked);
    top_layout->addWidget(btn_stop_);
    main_layout->addLayout(top_layout);

    // === CONTROL Group ===
    QGroupBox* control_group = new QGroupBox("CONTROL");
    QVBoxLayout* control_layout = new QVBoxLayout(control_group);

    QHBoxLayout* row1 = new QHBoxLayout();
    btn_enable_ = new QPushButton("Enable");
    btn_disable_ = new QPushButton("Disable");
    btn_reconnect_ = new QPushButton("Reconnect");
    btn_clear_error_ = new QPushButton("Clear Err");
    connect(btn_enable_, &QPushButton::clicked, this, &PiperGraspPanel::onEnableClicked);
    connect(btn_disable_, &QPushButton::clicked, this, &PiperGraspPanel::onDisableClicked);
    connect(btn_reconnect_, &QPushButton::clicked, this, &PiperGraspPanel::onReconnectClicked);
    connect(btn_clear_error_, &QPushButton::clicked, this, &PiperGraspPanel::onClearErrorClicked);
    row1->addWidget(btn_enable_);
    row1->addWidget(btn_disable_);
    row1->addWidget(btn_reconnect_);
    row1->addWidget(btn_clear_error_);
    control_layout->addLayout(row1);

    QHBoxLayout* row2 = new QHBoxLayout();
    btn_go_ready_ = new QPushButton("Go Ready");
    btn_go_zero_ = new QPushButton("Go Zero");
    btn_open_gripper_ = new QPushButton("Open Gripper");
    connect(btn_go_ready_, &QPushButton::clicked, this, &PiperGraspPanel::onGoReadyClicked);
    connect(btn_go_zero_, &QPushButton::clicked, this, &PiperGraspPanel::onGoZeroClicked);
    connect(btn_open_gripper_, &QPushButton::clicked, this, &PiperGraspPanel::onOpenGripperClicked);
    row2->addWidget(btn_go_ready_);
    row2->addWidget(btn_go_zero_);
    row2->addWidget(btn_open_gripper_);
    control_layout->addLayout(row2);

    main_layout->addWidget(control_group);

    // === GRASP Group ===
    QGroupBox* grasp_group = new QGroupBox("GRASP");
    QVBoxLayout* grasp_layout = new QVBoxLayout(grasp_group);

    // Prompt input
    QHBoxLayout* prompt_layout = new QHBoxLayout();
    prompt_layout->addWidget(new QLabel("Prompt:"));
    edit_prompt_ = new QLineEdit();
    edit_prompt_->setPlaceholderText("bottle.cup.toy.box.pen.key");
    prompt_layout->addWidget(edit_prompt_);
    grasp_layout->addLayout(prompt_layout);

    // Observe button
    btn_observe_ = new QPushButton("Observe");
    btn_observe_->setStyleSheet(
        "QPushButton { background-color: #6666CC; color: white; font-weight: bold; font-size: 14px; }"
        "QPushButton:hover { background-color: #8888EE; }"
        "QPushButton:disabled { background-color: #808080; }");
    btn_observe_->setMinimumHeight(40);
    connect(btn_observe_, &QPushButton::clicked, this, &PiperGraspPanel::onObserveClicked);
    grasp_layout->addWidget(btn_observe_);

    // Let's Grasp button + Cancel
    QHBoxLayout* grasp_btn_layout = new QHBoxLayout();
    btn_lets_grasp_ = new QPushButton("LET'S GRASP");
    btn_lets_grasp_->setStyleSheet(
        "QPushButton { background-color: #00AA00; color: white; font-weight: bold; font-size: 14px; }"
        "QPushButton:hover { background-color: #00CC00; }"
        "QPushButton:disabled { background-color: #808080; }");
    btn_lets_grasp_->setMinimumHeight(40);
    connect(btn_lets_grasp_, &QPushButton::clicked, this, &PiperGraspPanel::onLetsGraspClicked);
    grasp_btn_layout->addWidget(btn_lets_grasp_);
    btn_cancel_ = new QPushButton("Cancel");
    btn_cancel_->setEnabled(false);
    connect(btn_cancel_, &QPushButton::clicked, this, &PiperGraspPanel::onCancelClicked);
    grasp_btn_layout->addWidget(btn_cancel_);
    grasp_layout->addLayout(grasp_btn_layout);

    // Auto Clear button + End Clear
    QHBoxLayout* auto_clear_btn_layout = new QHBoxLayout();
    btn_auto_clear_ = new QPushButton("AUTO CLEAR");
    btn_auto_clear_->setStyleSheet(
        "QPushButton { background-color: #0066CC; color: white; font-weight: bold; font-size: 14px; }"
        "QPushButton:hover { background-color: #0088FF; }"
        "QPushButton:disabled { background-color: #808080; }");
    btn_auto_clear_->setMinimumHeight(40);
    connect(btn_auto_clear_, &QPushButton::clicked, this, &PiperGraspPanel::onAutoClearClicked);
    auto_clear_btn_layout->addWidget(btn_auto_clear_);

    btn_end_clear_ = new QPushButton("END CLEAR");
    btn_end_clear_->setEnabled(false);
    btn_end_clear_->setStyleSheet(
        "QPushButton { background-color: #CC6600; color: white; font-weight: bold; font-size: 14px; }"
        "QPushButton:hover { background-color: #FF8800; }"
        "QPushButton:disabled { background-color: #808080; }");
    btn_end_clear_->setMinimumHeight(40);
    connect(btn_end_clear_, &QPushButton::clicked, this, &PiperGraspPanel::onEndClearClicked);
    auto_clear_btn_layout->addWidget(btn_end_clear_);
    grasp_layout->addLayout(auto_clear_btn_layout);

    // Progress bar + status
    QHBoxLayout* progress_layout = new QHBoxLayout();
    bar_progress_ = new QProgressBar();
    bar_progress_->setRange(0, 100);
    bar_progress_->setValue(0);
    progress_layout->addWidget(bar_progress_);
    label_grasp_status_ = new QLabel("IDLE");
    label_grasp_status_->setMinimumWidth(80);
    progress_layout->addWidget(label_grasp_status_);
    grasp_layout->addLayout(progress_layout);

    // Last status
    label_last_status_ = new QLabel("");
    label_last_status_->setStyleSheet("color: gray;");
    grasp_layout->addWidget(label_last_status_);

    // Stage info
    label_stage_info_ = new QLabel("");
    label_stage_info_->setStyleSheet("font-family: monospace; background-color: #F0F0F0; padding: 5px; border: 1px solid #CCCCCC;");
    label_stage_info_->setWordWrap(true);
    label_stage_info_->setMinimumHeight(60);
    grasp_layout->addWidget(label_stage_info_);

    // Error display
    label_error_display_ = new QLabel("");
    label_error_display_->setStyleSheet("font-family: monospace; color: red; background-color: #FFE0E0; padding: 5px; border: 2px solid red; font-weight: bold;");
    label_error_display_->setWordWrap(true);
    label_error_display_->setVisible(false);
    label_error_display_->setMinimumHeight(40);
    grasp_layout->addWidget(label_error_display_);

    main_layout->addWidget(grasp_group);

    // === AUTO CLEAR STATISTICS Group ===
    group_statistics_ = new QGroupBox("AUTO CLEAR STATISTICS");
    QVBoxLayout* stats_layout = new QVBoxLayout(group_statistics_);

    btn_stats_toggle_ = new QPushButton("▼ Statistics");
    btn_stats_toggle_->setStyleSheet("text-align: left; font-weight: bold;");
    btn_stats_toggle_->setFlat(true);
    connect(btn_stats_toggle_, &QPushButton::clicked, this, &PiperGraspPanel::onStatisticsPanelToggled);
    stats_layout->addWidget(btn_stats_toggle_);

    label_cycle_count_ = new QLabel("Cycles: 0");
    label_success_count_ = new QLabel("Success: 0");
    label_fail_count_ = new QLabel("Failed: 0");
    label_success_rate_ = new QLabel("Rate: 0%");
    label_object_types_ = new QLabel("Objects: -");
    label_duration_ = new QLabel("Duration: 00:00");
    label_avg_time_ = new QLabel("Avg: 0s/cycle");

    label_cycle_count_->setStyleSheet("font-family: monospace;");
    label_success_count_->setStyleSheet("font-family: monospace; color: green;");
    label_fail_count_->setStyleSheet("font-family: monospace; color: red;");
    label_success_rate_->setStyleSheet("font-family: monospace; font-weight: bold;");
    label_object_types_->setStyleSheet("font-family: monospace;");
    label_duration_->setStyleSheet("font-family: monospace;");
    label_avg_time_->setStyleSheet("font-family: monospace;");

    // Add all labels to layout immediately (they own the widgets now)
    stats_layout->addWidget(label_cycle_count_);
    stats_layout->addWidget(label_success_count_);
    stats_layout->addWidget(label_fail_count_);
    stats_layout->addWidget(label_success_rate_);
    stats_layout->addWidget(label_object_types_);
    stats_layout->addWidget(label_duration_);
    stats_layout->addWidget(label_avg_time_);

    // Initially hide stats labels (collapsed state)
    label_cycle_count_->setVisible(false);
    label_success_count_->setVisible(false);
    label_fail_count_->setVisible(false);
    label_success_rate_->setVisible(false);
    label_object_types_->setVisible(false);
    label_duration_->setVisible(false);
    label_avg_time_->setVisible(false);

    group_statistics_->setVisible(false);
    main_layout->addWidget(group_statistics_);

    // === ARM DETAILS Group ===
    group_details_ = new QGroupBox("ARM DETAILS");
    QVBoxLayout* details_layout = new QVBoxLayout(group_details_);

    // Mode + Motion + Error code
    QHBoxLayout* status_row = new QHBoxLayout();
    status_row->addWidget(new QLabel("Mode:"));
    label_mode_ = new QLabel("--");
    label_mode_->setStyleSheet("font-weight: bold;");
    status_row->addWidget(label_mode_);
    status_row->addStretch();
    status_row->addWidget(new QLabel("Motion:"));
    label_motion_ = new QLabel("--");
    label_motion_->setStyleSheet("font-weight: bold;");
    status_row->addWidget(label_motion_);
    status_row->addStretch();
    status_row->addWidget(new QLabel("Err:"));
    label_err_code_ = new QLabel("0x0000");
    status_row->addWidget(label_err_code_);
    details_layout->addLayout(status_row);

    // Joint Collision Status
    QHBoxLayout* limits_row = new QHBoxLayout();
    limits_row->addWidget(new QLabel("Coll:"));
    for (int i = 0; i < 6; i++) {
        label_joint_limits_[i] = new QLabel(QString("%1").arg(i+1));
        label_joint_limits_[i]->setAlignment(Qt::AlignCenter);
        label_joint_limits_[i]->setMinimumWidth(20);
        label_joint_limits_[i]->setStyleSheet("background-color: #00AA00; color: white; border-radius: 3px;");
        limits_row->addWidget(label_joint_limits_[i]);
    }
    limits_row->addStretch();
    details_layout->addLayout(limits_row);

    // Joint Comms
    QHBoxLayout* comms_row = new QHBoxLayout();
    comms_row->addWidget(new QLabel("Comm:"));
    for (int i = 0; i < 6; i++) {
        label_joint_comms_[i] = new QLabel(QString("%1").arg(i+1));
        label_joint_comms_[i]->setAlignment(Qt::AlignCenter);
        label_joint_comms_[i]->setMinimumWidth(20);
        label_joint_comms_[i]->setStyleSheet("background-color: #00AA00; color: white; border-radius: 3px;");
        comms_row->addWidget(label_joint_comms_[i]);
    }
    comms_row->addStretch();
    details_layout->addLayout(comms_row);

    // Position (Gripper Center)
    QHBoxLayout* pos_row = new QHBoxLayout();
    pos_row->addWidget(new QLabel("GC:"));
    label_position_ = new QLabel("X:-- Y:-- Z:-- R:-- P:-- Y:--");
    label_position_->setStyleSheet("font-family: monospace;");
    label_position_->setToolTip("Gripper Center position (mm) and orientation (deg)");
    pos_row->addWidget(label_position_);
    details_layout->addLayout(pos_row);

    // Joints
    QHBoxLayout* joints_row = new QHBoxLayout();
    joints_row->addWidget(new QLabel("Joint:"));
    label_joints_ = new QLabel("[--, --, --, --, --, --]");
    label_joints_->setStyleSheet("font-family: monospace;");
    joints_row->addWidget(label_joints_);
    details_layout->addLayout(joints_row);

    // Gripper
    QHBoxLayout* gripper_row = new QHBoxLayout();
    gripper_row->addWidget(new QLabel("Gripper:"));
    bar_gripper_ = new QProgressBar();
    bar_gripper_->setRange(0, static_cast<int>(GRIPPER_MAX_MM));
    bar_gripper_->setValue(0);
    bar_gripper_->setTextVisible(false);
    gripper_row->addWidget(bar_gripper_);
    label_gripper_value_ = new QLabel("0.0 mm");
    label_gripper_value_->setMinimumWidth(60);
    gripper_row->addWidget(label_gripper_value_);
    details_layout->addLayout(gripper_row);

    main_layout->addWidget(group_details_);
    main_layout->addStretch();

    // Finish scroll area setup
    scroll_area->setWidget(scroll_content);
    outer_layout->addWidget(scroll_area);

    // Set reasonable size policy for RViz2 dock
    setMinimumWidth(280);
}

void PiperGraspPanel::setupRos()
{
    if (!node_) return;

    // Subscribers
    sub_status_ = node_->create_subscription<piper_msgs::msg::PiperStatus>(
        "/piper/status", 1,
        std::bind(&PiperGraspPanel::statusCallback, this, std::placeholders::_1));

    sub_arm_status_ = node_->create_subscription<piper_msgs::msg::PiperStatusMsg>(
        "/arm_status", 1,
        std::bind(&PiperGraspPanel::armStatusCallback, this, std::placeholders::_1));

    // Service clients
    srv_enable_ = node_->create_client<piper_msgs::srv::EnableEnhanced>("/piper/enable");
    srv_go_ready_ = node_->create_client<piper_msgs::srv::GoReady>("/piper/go_ready");
    srv_go_zero_ = node_->create_client<piper_msgs::srv::GoZero>("/go_zero_srv");
    srv_set_gripper_ = node_->create_client<piper_msgs::srv::SetGripper>("/piper/set_gripper");
    srv_stop_ = node_->create_client<std_srvs::srv::Trigger>("/stop_srv");
    srv_observe_ = node_->create_client<piper_msgs::srv::Observe>("/piper/observe");

    // Action clients
    pick_client_ = rclcpp_action::create_client<PickAction>(node_, "/piper/pick");
    place_client_ = rclcpp_action::create_client<PlaceAction>(node_, "/piper/place");
}

// === ROS2 Callbacks ===

void PiperGraspPanel::statusCallback(const piper_msgs::msg::PiperStatus::SharedPtr msg)
{
    int state = 0;
    if (msg->connected && msg->enabled) state = 2;
    else if (msg->connected) state = 1;
    connection_state_.store(state);

    Q_EMIT statusUpdated(msg->connected, msg->enabled,
                         QString::fromStdString(msg->error_state),
                         toQVector(msg->gripper_center),
                         toQVector(msg->joints),
                         msg->gripper_position);
}

void PiperGraspPanel::armStatusCallback(const piper_msgs::msg::PiperStatusMsg::SharedPtr msg)
{
    QVector<bool> limits, comms;
    limits << msg->joint_1_angle_limit << msg->joint_2_angle_limit
           << msg->joint_3_angle_limit << msg->joint_4_angle_limit
           << msg->joint_5_angle_limit << msg->joint_6_angle_limit;
    comms << msg->communication_status_joint_1 << msg->communication_status_joint_2
          << msg->communication_status_joint_3 << msg->communication_status_joint_4
          << msg->communication_status_joint_5 << msg->communication_status_joint_6;

    Q_EMIT armStatusUpdated(msg->ctrl_mode, msg->motion_status, msg->err_code, limits, comms);
}

// === Signal Handlers ===

void PiperGraspPanel::handleStatusUpdated(bool connected, bool enabled,
                                          const QString& error_state,
                                          const QVector<double>& position,
                                          const QVector<double>& joints,
                                          double gripper_pos)
{
    updateConnectionDisplay(connected, enabled);
    current_position_ = position;

    label_error_->setText(QString("Error: %1").arg(error_state));
    if (error_state == "NORMAL") {
        label_error_->setStyleSheet("color: green;");
    } else if (error_state == "WARNING") {
        label_error_->setStyleSheet("color: orange;");
    } else {
        label_error_->setStyleSheet("color: red; font-weight: bold;");
    }

    if (position.size() >= 6) {
        label_position_->setText(QString("X:%1 Y:%2 Z:%3 R:%4 P:%5 Y:%6")
            .arg(position[0], 0, 'f', 0)
            .arg(position[1], 0, 'f', 0)
            .arg(position[2], 0, 'f', 0)
            .arg(position[3], 0, 'f', 0)
            .arg(position[4], 0, 'f', 0)
            .arg(position[5], 0, 'f', 0));
    }

    if (joints.size() >= 6) {
        label_joints_->setText(QString("[%1, %2, %3, %4, %5, %6]")
            .arg(joints[0], 0, 'f', 1)
            .arg(joints[1], 0, 'f', 1)
            .arg(joints[2], 0, 'f', 1)
            .arg(joints[3], 0, 'f', 1)
            .arg(joints[4], 0, 'f', 1)
            .arg(joints[5], 0, 'f', 1));
    }

    bar_gripper_->setValue(static_cast<int>(gripper_pos));
    label_gripper_value_->setText(QString("%1 mm").arg(gripper_pos, 0, 'f', 1));
}

void PiperGraspPanel::handleArmStatusUpdated(uint8_t ctrl_mode, uint8_t motion_status,
                                             int64_t err_code,
                                             const QVector<bool>& joint_limits,
                                             const QVector<bool>& joint_comms)
{
    label_mode_->setText(getModeText(ctrl_mode));
    label_motion_->setText(getMotionText(motion_status));
    label_err_code_->setText(QString("0x%1").arg(err_code, 4, 16, QChar('0')).toUpper());

    if (err_code != 0) {
        label_err_code_->setStyleSheet("color: red; font-weight: bold;");
    } else {
        label_err_code_->setStyleSheet("");
    }

    for (int i = 0; i < 6 && i < joint_limits.size(); i++) {
        if (joint_limits[i]) {
            label_joint_limits_[i]->setStyleSheet("background-color: #FF0000; color: white; border-radius: 3px;");
        } else {
            label_joint_limits_[i]->setStyleSheet("background-color: #00AA00; color: white; border-radius: 3px;");
        }
    }

    for (int i = 0; i < 6 && i < joint_comms.size(); i++) {
        if (joint_comms[i]) {
            label_joint_comms_[i]->setStyleSheet("background-color: #00AA00; color: white; border-radius: 3px;");
        } else {
            label_joint_comms_[i]->setStyleSheet("background-color: #FF0000; color: white; border-radius: 3px;");
        }
    }
}

void PiperGraspPanel::handleServiceResult(const QString& name, bool success, const QString& msg)
{
    setLastStatus(QString("%1: %2").arg(name).arg(msg), success);
}

void PiperGraspPanel::handleObserveResult(bool success, const QString& category, double score,
                                          const QString& error_msg)
{
    LetsGraspState state = lets_grasp_state_.load();

    if (success) {
        observe_valid_.store(true);
        observe_category_ = category;
        observe_score_ = score;
        setLastStatus(QString("Observe: %1 (%2)").arg(category).arg(score, 0, 'f', 2), true);

        QString info = QString("[OBSERVE RESULT]\n");
        if (!current_position_.isEmpty() && current_position_.size() >= 6) {
            info += QString("Current Pose: X:%1 Y:%2 Z:%3\n")
                    .arg(current_position_[0], 0, 'f', 1)
                    .arg(current_position_[1], 0, 'f', 1)
                    .arg(current_position_[2], 0, 'f', 1);
        }
        info += QString("Detected: %1 (Score: %2)")
                .arg(category).arg(score, 0, 'f', 3);
        label_stage_info_->setText(info);

        if (state == LetsGraspState::OBSERVE_ONLY) {
            QTimer::singleShot(2000, this, [this]() {
                if (!panel_alive_.load()) return;
                if (lets_grasp_state_.load() == LetsGraspState::OBSERVE_ONLY) {
                    lets_grasp_state_.store(LetsGraspState::IDLE);
                    updateLetsGraspUI();
                }
            });
            return;
        }

        if (auto_clear_mode_.load()) {
            auto_consecutive_empty_observes_ = 0;
        }

        // Auto-trigger Pick
        if (state == LetsGraspState::OBSERVING || state == LetsGraspState::AUTO_OBSERVING) {
            if (state == LetsGraspState::AUTO_OBSERVING) {
                lets_grasp_state_.store(LetsGraspState::AUTO_PICKING);
            } else {
                lets_grasp_state_.store(LetsGraspState::PICKING);
            }
            updateLetsGraspUI();

            // Send pick goal
            if (pick_client_ && pick_client_->wait_for_action_server(std::chrono::seconds(1))) {
                auto goal = PickAction::Goal();
                goal.use_last_observe = true;
                goal.speed = 30;
                goal.return_to_ready = false;

                auto send_goal_options = rclcpp_action::Client<PickAction>::SendGoalOptions();
                send_goal_options.goal_response_callback =
                    std::bind(&PiperGraspPanel::pickGoalResponseCallback, this, std::placeholders::_1);
                send_goal_options.feedback_callback =
                    std::bind(&PiperGraspPanel::pickFeedbackCallback, this, std::placeholders::_1, std::placeholders::_2);
                send_goal_options.result_callback =
                    std::bind(&PiperGraspPanel::pickResultCallback, this, std::placeholders::_1);

                pick_client_->async_send_goal(goal, send_goal_options);
            }
        }
    } else {
        observe_valid_.store(false);
        setLastStatus(QString("Observe failed: %1").arg(error_msg), false);
        showErrorMessage(QString("[ERROR - OBSERVE]\nMessage: %1").arg(error_msg));

        if (state == LetsGraspState::OBSERVE_ONLY) {
            QTimer::singleShot(3000, this, [this]() {
                if (!panel_alive_.load()) return;
                if (lets_grasp_state_.load() == LetsGraspState::OBSERVE_ONLY) {
                    lets_grasp_state_.store(LetsGraspState::IDLE);
                    updateLetsGraspUI();
                }
            });
            return;
        }

        if (state == LetsGraspState::AUTO_OBSERVING) {
            auto_consecutive_empty_observes_++;
            auto_fail_count_++;
            updateAutoStatistics();
            checkAutoTerminationConditions();
            if (auto_clear_mode_.load()) {
                lets_grasp_state_.store(LetsGraspState::AUTO_WAITING);
                updateLetsGraspUI();
                auto_cycle_timer_->start(AUTO_CYCLE_DELAY_FAILURE_MS);
            }
        } else if (state == LetsGraspState::OBSERVING) {
            lets_grasp_state_.store(LetsGraspState::ERROR);
            lets_grasp_error_msg_ = error_msg;
            updateLetsGraspUI();
            QTimer::singleShot(3000, this, [this]() {
                if (!panel_alive_.load()) return;
                lets_grasp_state_.store(LetsGraspState::IDLE);
                updateLetsGraspUI();
            });
        }
    }
}

void PiperGraspPanel::handlePickProgress(double progress, const QString& step)
{
    LetsGraspState state = lets_grasp_state_.load();
    if (state == LetsGraspState::PICKING || state == LetsGraspState::AUTO_PICKING) {
        bar_progress_->setValue(static_cast<int>(30 + progress * 35));
        label_grasp_status_->setText(step);
    }
}

void PiperGraspPanel::handlePickResult(bool success, const QString& message)
{
    if (success) {
        setLastStatus(QString("Pick: %1").arg(message), true);

        LetsGraspState state = lets_grasp_state_.load();
        if (state == LetsGraspState::PICKING || state == LetsGraspState::AUTO_PICKING) {
            if (state == LetsGraspState::AUTO_PICKING) {
                lets_grasp_state_.store(LetsGraspState::AUTO_PLACING);
                if (!observe_category_.isEmpty()) {
                    auto_object_counts_[observe_category_]++;
                }
            } else {
                lets_grasp_state_.store(LetsGraspState::PLACING);
            }
            updateLetsGraspUI();

            // Send place goal
            if (place_client_ && place_client_->wait_for_action_server(std::chrono::seconds(1))) {
                auto goal = PlaceAction::Goal();
                goal.use_default_place = true;
                goal.speed = 30;
                goal.return_to_ready = true;

                auto send_goal_options = rclcpp_action::Client<PlaceAction>::SendGoalOptions();
                send_goal_options.goal_response_callback =
                    std::bind(&PiperGraspPanel::placeGoalResponseCallback, this, std::placeholders::_1);
                send_goal_options.feedback_callback =
                    std::bind(&PiperGraspPanel::placeFeedbackCallback, this, std::placeholders::_1, std::placeholders::_2);
                send_goal_options.result_callback =
                    std::bind(&PiperGraspPanel::placeResultCallback, this, std::placeholders::_1);

                place_client_->async_send_goal(goal, send_goal_options);
            }
        }
    } else {
        setLastStatus(QString("Pick failed: %1").arg(message), false);
        showErrorMessage(QString("[ERROR - PICK]\nMessage: %1").arg(message));

        LetsGraspState state = lets_grasp_state_.load();

        if (state == LetsGraspState::AUTO_PICKING) {
            auto_fail_count_++;
            auto_consecutive_failures_++;
            updateAutoStatistics();
            onGoReadyClicked();
            checkAutoTerminationConditions();
            if (auto_clear_mode_.load()) {
                lets_grasp_state_.store(LetsGraspState::AUTO_WAITING);
                updateLetsGraspUI();
                auto_cycle_timer_->start(AUTO_CYCLE_DELAY_FAILURE_MS);
            }
        } else if (state == LetsGraspState::PICKING) {
            lets_grasp_state_.store(LetsGraspState::ERROR);
            lets_grasp_error_msg_ = message;
            updateLetsGraspUI();
            onGoReadyClicked();
            QTimer::singleShot(3000, this, [this]() {
                if (!panel_alive_.load()) return;
                lets_grasp_state_.store(LetsGraspState::IDLE);
                updateLetsGraspUI();
            });
        }
    }
}

void PiperGraspPanel::handlePlaceProgress(double progress, const QString& step)
{
    LetsGraspState state = lets_grasp_state_.load();
    if (state == LetsGraspState::PLACING || state == LetsGraspState::AUTO_PLACING) {
        bar_progress_->setValue(static_cast<int>(65 + progress * 35));
        label_grasp_status_->setText(step);
    }
}

void PiperGraspPanel::handlePlaceResult(bool success, const QString& message)
{
    if (success) {
        setLastStatus(QString("Place: %1").arg(message), true);

        LetsGraspState state = lets_grasp_state_.load();

        if (state == LetsGraspState::AUTO_PLACING) {
            auto_success_count_++;
            auto_consecutive_failures_ = 0;
            updateAutoStatistics();

            if (end_clear_requested_.load()) {
                auto_clear_mode_.store(false);
                lets_grasp_state_.store(LetsGraspState::IDLE);
                updateLetsGraspUI();

                btn_auto_clear_->setEnabled(true);
                btn_end_clear_->setEnabled(false);
                btn_end_clear_->setText("END CLEAR");
                btn_lets_grasp_->setEnabled(true);

                setLastStatus("Auto Clear ended (user request)", true);
            } else {
                startNextAutoCycle();
            }
        } else if (state == LetsGraspState::PLACING) {
            lets_grasp_state_.store(LetsGraspState::COMPLETE);
            updateLetsGraspUI();
            QTimer::singleShot(3000, this, [this]() {
                if (!panel_alive_.load()) return;
                lets_grasp_state_.store(LetsGraspState::IDLE);
                updateLetsGraspUI();
            });
        }
    } else {
        setLastStatus(QString("Place failed: %1").arg(message), false);
        showErrorMessage(QString("[ERROR - PLACE]\nMessage: %1").arg(message));

        if (lets_grasp_state_.load() == LetsGraspState::PLACING) {
            lets_grasp_state_.store(LetsGraspState::ERROR);
            lets_grasp_error_msg_ = message;
            updateLetsGraspUI();
            QTimer::singleShot(3000, this, [this]() {
                if (!panel_alive_.load()) return;
                lets_grasp_state_.store(LetsGraspState::IDLE);
                updateLetsGraspUI();
            });
        }
    }
}

// === Action Callbacks ===

void PiperGraspPanel::pickGoalResponseCallback(
    const rclcpp_action::ClientGoalHandle<PickAction>::SharedPtr& goal_handle)
{
    if (!goal_handle) {
        Q_EMIT pickResult(false, "Goal rejected");
    } else {
        current_pick_goal_ = goal_handle;
    }
}

void PiperGraspPanel::pickFeedbackCallback(
    PickGoalHandle::SharedPtr,
    const std::shared_ptr<const PickAction::Feedback> feedback)
{
    Q_EMIT pickProgress(feedback->progress, QString::fromStdString(feedback->step_name));
}

void PiperGraspPanel::pickResultCallback(
    const PickGoalHandle::WrappedResult& result)
{
    current_pick_goal_.reset();

    if (result.code == rclcpp_action::ResultCode::SUCCEEDED) {
        QString msg = result.result->success ?
            QString("%1 (%2ms)").arg(QString::fromStdString(result.result->category))
                                .arg(result.result->execution_time_ms, 0, 'f', 0) :
            QString::fromStdString(result.result->error_message);
        Q_EMIT pickResult(result.result->success, msg);
    } else {
        Q_EMIT pickResult(false, "Action failed");
    }
}

void PiperGraspPanel::placeGoalResponseCallback(
    const rclcpp_action::ClientGoalHandle<PlaceAction>::SharedPtr& goal_handle)
{
    if (!goal_handle) {
        Q_EMIT placeResult(false, "Goal rejected");
    } else {
        current_place_goal_ = goal_handle;
    }
}

void PiperGraspPanel::placeFeedbackCallback(
    PlaceGoalHandle::SharedPtr,
    const std::shared_ptr<const PlaceAction::Feedback> feedback)
{
    Q_EMIT placeProgress(feedback->progress, QString::fromStdString(feedback->step_name));
}

void PiperGraspPanel::placeResultCallback(
    const PlaceGoalHandle::WrappedResult& result)
{
    current_place_goal_.reset();

    if (result.code == rclcpp_action::ResultCode::SUCCEEDED) {
        QString msg = result.result->success ?
            QString("%1ms").arg(result.result->execution_time_ms, 0, 'f', 0) :
            QString::fromStdString(result.result->error_message);
        Q_EMIT placeResult(result.result->success, msg);
    } else {
        Q_EMIT placeResult(false, "Action failed");
    }
}

// === Button Handlers ===

void PiperGraspPanel::onStopClicked()
{
    auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
    auto future = srv_stop_->async_send_request(request);

    // Cancel any running actions
    if (current_pick_goal_) {
        pick_client_->async_cancel_goal(current_pick_goal_);
    }
    if (current_place_goal_) {
        place_client_->async_cancel_goal(current_place_goal_);
    }

    showErrorMessage("[ERROR - EMERGENCY STOP]\nMessage: User triggered emergency stop");

    if (lets_grasp_state_.load() != LetsGraspState::IDLE) {
        lets_grasp_state_.store(LetsGraspState::ERROR);
        lets_grasp_error_msg_ = "Emergency stop";
        updateLetsGraspUI();
        QTimer::singleShot(2000, this, [this]() {
            if (!panel_alive_.load()) return;
            lets_grasp_state_.store(LetsGraspState::IDLE);
            updateLetsGraspUI();
        });
    }
}

void PiperGraspPanel::onEnableClicked()
{
    std::thread([this]() {
        if (!panel_alive_.load() || !node_) return;
        auto request = std::make_shared<piper_msgs::srv::EnableEnhanced::Request>();
        request->action = piper_msgs::srv::EnableEnhanced::Request::ACTION_ENABLE;
        request->go_zero = false;
        auto future = srv_enable_->async_send_request(request);
        if (future.wait_for(std::chrono::seconds(10)) == std::future_status::ready) {
            auto response = future.get();
            if (panel_alive_.load()) {
                Q_EMIT serviceResult("Enable", response->success, QString::fromStdString(response->message));
            }
        } else {
            if (panel_alive_.load()) {
                Q_EMIT serviceResult("Enable", false, "Timeout");
            }
        }
    }).detach();
}

void PiperGraspPanel::onDisableClicked()
{
    std::thread([this]() {
        if (!panel_alive_.load() || !node_) return;
        auto request = std::make_shared<piper_msgs::srv::EnableEnhanced::Request>();
        request->action = piper_msgs::srv::EnableEnhanced::Request::ACTION_DISABLE;
        request->go_zero = false;
        auto future = srv_enable_->async_send_request(request);
        if (future.wait_for(std::chrono::seconds(10)) == std::future_status::ready) {
            auto response = future.get();
            if (panel_alive_.load()) {
                Q_EMIT serviceResult("Disable", response->success, QString::fromStdString(response->message));
            }
        } else {
            if (panel_alive_.load()) {
                Q_EMIT serviceResult("Disable", false, "Timeout");
            }
        }
    }).detach();
}

void PiperGraspPanel::onReconnectClicked()
{
    std::thread([this]() {
        if (!panel_alive_.load() || !node_) return;
        auto request = std::make_shared<piper_msgs::srv::EnableEnhanced::Request>();
        request->action = piper_msgs::srv::EnableEnhanced::Request::ACTION_RECONNECT;
        auto future = srv_enable_->async_send_request(request);
        if (future.wait_for(std::chrono::seconds(10)) == std::future_status::ready) {
            auto response = future.get();
            if (panel_alive_.load()) {
                Q_EMIT serviceResult("Reconnect", response->success, QString::fromStdString(response->message));
            }
        } else {
            if (panel_alive_.load()) {
                Q_EMIT serviceResult("Reconnect", false, "Timeout");
            }
        }
    }).detach();
}

void PiperGraspPanel::onClearErrorClicked()
{
    std::thread([this]() {
        if (!panel_alive_.load() || !node_) return;
        auto request = std::make_shared<piper_msgs::srv::EnableEnhanced::Request>();
        request->action = piper_msgs::srv::EnableEnhanced::Request::ACTION_CLEAN_ERROR;
        auto future = srv_enable_->async_send_request(request);
        if (future.wait_for(std::chrono::seconds(10)) == std::future_status::ready) {
            auto response = future.get();
            if (panel_alive_.load()) {
                Q_EMIT serviceResult("Clear Error", response->success, QString::fromStdString(response->message));
            }
        } else {
            if (panel_alive_.load()) {
                Q_EMIT serviceResult("Clear Error", false, "Timeout");
            }
        }
    }).detach();
}

void PiperGraspPanel::onGoReadyClicked()
{
    std::thread([this]() {
        if (!panel_alive_.load() || !node_) return;
        auto request = std::make_shared<piper_msgs::srv::GoReady::Request>();
        request->speed = 30;
        request->open_gripper = false;
        auto future = srv_go_ready_->async_send_request(request);
        if (future.wait_for(std::chrono::seconds(30)) == std::future_status::ready) {
            auto response = future.get();
            if (panel_alive_.load()) {
                Q_EMIT serviceResult("Go Ready", response->success, QString::fromStdString(response->message));
            }
        } else {
            if (panel_alive_.load()) {
                Q_EMIT serviceResult("Go Ready", false, "Timeout");
            }
        }
    }).detach();
}

void PiperGraspPanel::onGoZeroClicked()
{
    std::thread([this]() {
        if (!panel_alive_.load() || !node_) return;
        auto request = std::make_shared<piper_msgs::srv::GoZero::Request>();
        request->is_mit_mode = false;
        auto future = srv_go_zero_->async_send_request(request);
        if (future.wait_for(std::chrono::seconds(30)) == std::future_status::ready) {
            auto response = future.get();
            if (panel_alive_.load()) {
                Q_EMIT serviceResult("Go Zero", response->status, "Done");
            }
        } else {
            if (panel_alive_.load()) {
                Q_EMIT serviceResult("Go Zero", false, "Timeout");
            }
        }
    }).detach();
}

void PiperGraspPanel::onOpenGripperClicked()
{
    std::thread([this]() {
        if (!panel_alive_.load() || !node_) return;
        auto request = std::make_shared<piper_msgs::srv::SetGripper::Request>();
        request->position = GRIPPER_MAX_MM;
        request->speed = 500;
        request->wait = true;
        auto future = srv_set_gripper_->async_send_request(request);
        if (future.wait_for(std::chrono::seconds(10)) == std::future_status::ready) {
            auto response = future.get();
            if (panel_alive_.load()) {
                Q_EMIT serviceResult("Open Gripper", response->success, QString::fromStdString(response->message));
            }
        } else {
            if (panel_alive_.load()) {
                Q_EMIT serviceResult("Open Gripper", false, "Timeout");
            }
        }
    }).detach();
}

void PiperGraspPanel::onObserveClicked()
{
    LetsGraspState state = lets_grasp_state_.load();
    if (state != LetsGraspState::IDLE) {
        setLastStatus("Busy, cannot observe now", false);
        return;
    }

    int conn = connection_state_.load();
    if (conn == 0) {
        setLastStatus("Arm not connected", false);
        return;
    }
    if (conn == 1) {
        setLastStatus("Arm not enabled", false);
        return;
    }

    label_error_display_->setText("");
    label_error_display_->setVisible(false);
    current_target_grasp_.clear();

    QString prompt = edit_prompt_->text().trimmed();
    if (!prompt.isEmpty()) {
        current_prompt_ = prompt;
    }

    lets_grasp_state_.store(LetsGraspState::OBSERVE_ONLY);
    updateLetsGraspUI();

    setLastStatus(QString("Observing: %1").arg(current_prompt_), true);

    std::string prompt_str = current_prompt_.toStdString();

    std::thread([this, prompt_str]() {
        if (!panel_alive_.load() || !node_) return;

        // Go ready first
        auto ready_req = std::make_shared<piper_msgs::srv::GoReady::Request>();
        ready_req->speed = 30;
        ready_req->open_gripper = true;
        auto ready_future = srv_go_ready_->async_send_request(ready_req);
        if (ready_future.wait_for(std::chrono::seconds(30)) != std::future_status::ready ||
            !ready_future.get()->success) {
            if (panel_alive_.load()) {
                Q_EMIT observeResult(false, "", 0, "Go ready failed");
            }
            return;
        }

        if (!panel_alive_.load()) return;

        // Observe
        auto observe_req = std::make_shared<piper_msgs::srv::Observe::Request>();
        observe_req->prompt = prompt_str;
        observe_req->enable_cdm = true;
        auto observe_future = srv_observe_->async_send_request(observe_req);
        if (observe_future.wait_for(std::chrono::seconds(30)) == std::future_status::ready) {
            auto response = observe_future.get();
            if (panel_alive_.load()) {
                Q_EMIT observeResult(response->success,
                                     QString::fromStdString(response->category),
                                     response->score,
                                     QString::fromStdString(response->error_message));
            }
        } else {
            if (panel_alive_.load()) {
                Q_EMIT observeResult(false, "", 0, "Observe timeout");
            }
        }
    }).detach();
}

void PiperGraspPanel::onLetsGraspClicked()
{
    if (lets_grasp_state_.load() != LetsGraspState::IDLE) {
        setLastStatus("Already running", false);
        return;
    }

    label_error_display_->setText("");
    label_error_display_->setVisible(false);
    target_grasp_pose_.clear();

    int conn = connection_state_.load();
    if (conn == 0) {
        setLastStatus("Arm not connected", false);
        return;
    }
    if (conn == 1) {
        setLastStatus("Arm not enabled", false);
        return;
    }

    if (!pick_client_ || !pick_client_->wait_for_action_server(std::chrono::seconds(1))) {
        setLastStatus("Pick server not connected", false);
        return;
    }
    if (!place_client_ || !place_client_->wait_for_action_server(std::chrono::seconds(1))) {
        setLastStatus("Place server not connected", false);
        return;
    }

    QString prompt = edit_prompt_->text().trimmed();
    if (!prompt.isEmpty()) {
        current_prompt_ = prompt;
    }

    lets_grasp_state_.store(LetsGraspState::OBSERVING);
    lets_grasp_error_msg_.clear();
    updateLetsGraspUI();

    setLastStatus(QString("Starting: %1").arg(current_prompt_), true);

    std::string prompt_str = current_prompt_.toStdString();

    std::thread([this, prompt_str]() {
        if (!panel_alive_.load() || !node_) return;

        // Go ready first
        auto ready_req = std::make_shared<piper_msgs::srv::GoReady::Request>();
        ready_req->speed = 30;
        ready_req->open_gripper = true;
        auto ready_future = srv_go_ready_->async_send_request(ready_req);
        if (ready_future.wait_for(std::chrono::seconds(30)) != std::future_status::ready ||
            !ready_future.get()->success) {
            if (panel_alive_.load()) {
                Q_EMIT observeResult(false, "", 0, "Go ready failed");
            }
            return;
        }

        if (!panel_alive_.load()) return;
        if (lets_grasp_state_.load() == LetsGraspState::CANCELLING) {
            Q_EMIT observeResult(false, "", 0, "Cancelled");
            return;
        }

        // Observe
        auto observe_req = std::make_shared<piper_msgs::srv::Observe::Request>();
        observe_req->prompt = prompt_str;
        observe_req->enable_cdm = true;
        auto observe_future = srv_observe_->async_send_request(observe_req);
        if (observe_future.wait_for(std::chrono::seconds(30)) == std::future_status::ready) {
            auto response = observe_future.get();
            if (panel_alive_.load()) {
                if (response->success && response->point3d_base.size() >= 3) {
                    current_target_grasp_.clear();
                    for (float val : response->point3d_base) {
                        current_target_grasp_.append(static_cast<double>(val));
                    }
                }
                Q_EMIT observeResult(response->success,
                                     QString::fromStdString(response->category),
                                     response->score,
                                     QString::fromStdString(response->error_message));
            }
        } else {
            if (panel_alive_.load()) {
                Q_EMIT observeResult(false, "", 0, "Observe timeout");
            }
        }
    }).detach();
}

void PiperGraspPanel::onCancelClicked()
{
    LetsGraspState state = lets_grasp_state_.load();
    if (state == LetsGraspState::IDLE ||
        state == LetsGraspState::COMPLETE ||
        state == LetsGraspState::ERROR) {
        return;
    }

    lets_grasp_state_.store(LetsGraspState::CANCELLING);
    updateLetsGraspUI();

    if (current_pick_goal_) {
        pick_client_->async_cancel_goal(current_pick_goal_);
    }
    if (current_place_goal_) {
        place_client_->async_cancel_goal(current_place_goal_);
    }

    onGoReadyClicked();

    setLastStatus("Cancelled", false);

    QTimer::singleShot(1000, this, [this]() {
        if (!panel_alive_.load()) return;
        lets_grasp_state_.store(LetsGraspState::IDLE);
        updateLetsGraspUI();
    });
}

// === Auto Clear Functions ===

void PiperGraspPanel::onAutoClearClicked()
{
    if (lets_grasp_state_.load() != LetsGraspState::IDLE) {
        setLastStatus("Already running", false);
        return;
    }

    int conn = connection_state_.load();
    if (conn == 0) {
        setLastStatus("Arm not connected", false);
        return;
    }
    if (conn == 1) {
        setLastStatus("Arm not enabled", false);
        return;
    }

    resetAutoStatistics();

    auto_clear_mode_.store(true);
    end_clear_requested_.store(false);

    group_statistics_->setVisible(true);
    statistics_panel_expanded_ = true;
    btn_stats_toggle_->setText("▲ Statistics");
    // Show all stats labels
    label_cycle_count_->setVisible(true);
    label_success_count_->setVisible(true);
    label_fail_count_->setVisible(true);
    label_success_rate_->setVisible(true);
    label_object_types_->setVisible(true);
    label_duration_->setVisible(true);
    label_avg_time_->setVisible(true);

    btn_auto_clear_->setEnabled(false);
    btn_end_clear_->setEnabled(true);
    btn_lets_grasp_->setEnabled(false);

    setLastStatus("Auto Clear started", true);

    startNextAutoCycle();
}

void PiperGraspPanel::onEndClearClicked()
{
    end_clear_requested_.store(true);
    btn_end_clear_->setEnabled(false);
    btn_end_clear_->setText("ENDING...");
    setLastStatus("Ending Auto Clear...", true);
}

void PiperGraspPanel::onStatisticsPanelToggled()
{
    statistics_panel_expanded_ = !statistics_panel_expanded_;

    // Toggle visibility of statistics labels
    bool show = statistics_panel_expanded_;
    label_cycle_count_->setVisible(show);
    label_success_count_->setVisible(show);
    label_fail_count_->setVisible(show);
    label_success_rate_->setVisible(show);
    label_object_types_->setVisible(show);
    label_duration_->setVisible(show);
    label_avg_time_->setVisible(show);

    btn_stats_toggle_->setText(show ? "▲ Statistics" : "▼ Statistics");
}

void PiperGraspPanel::startNextAutoCycle()
{
    if (!panel_alive_.load()) return;

    if (end_clear_requested_.load()) {
        auto_clear_mode_.store(false);
        lets_grasp_state_.store(LetsGraspState::IDLE);
        updateLetsGraspUI();

        btn_auto_clear_->setEnabled(true);
        btn_end_clear_->setEnabled(false);
        btn_end_clear_->setText("END CLEAR");
        btn_lets_grasp_->setEnabled(true);

        setLastStatus("Auto Clear ended", true);
        return;
    }

    checkAutoTerminationConditions();
    if (!auto_clear_mode_.load()) {
        return;
    }

    auto_cycle_count_++;
    auto_cycle_start_time_ = QTime::currentTime();
    updateAutoStatistics();

    lets_grasp_state_.store(LetsGraspState::AUTO_OBSERVING);
    updateLetsGraspUI();

    QString prompt = edit_prompt_->text().trimmed();
    if (!prompt.isEmpty()) {
        current_prompt_ = prompt;
    }

    setLastStatus(QString("Auto Cycle %1: Starting").arg(auto_cycle_count_), true);

    std::string prompt_str = current_prompt_.toStdString();

    std::thread([this, prompt_str]() {
        if (!panel_alive_.load() || !node_) return;

        // Go ready first
        auto ready_req = std::make_shared<piper_msgs::srv::GoReady::Request>();
        ready_req->speed = 30;
        ready_req->open_gripper = true;
        auto ready_future = srv_go_ready_->async_send_request(ready_req);
        if (ready_future.wait_for(std::chrono::seconds(30)) != std::future_status::ready ||
            !ready_future.get()->success) {
            if (panel_alive_.load()) {
                Q_EMIT observeResult(false, "", 0, "Go ready failed");
            }
            return;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(CAMERA_SETTLE_TIME_MS));

        if (!panel_alive_.load()) return;
        if (end_clear_requested_.load()) {
            Q_EMIT observeResult(false, "", 0, "Cancelled");
            return;
        }

        // Observe
        auto observe_req = std::make_shared<piper_msgs::srv::Observe::Request>();
        observe_req->prompt = prompt_str;
        observe_req->enable_cdm = true;
        auto observe_future = srv_observe_->async_send_request(observe_req);
        if (observe_future.wait_for(std::chrono::seconds(30)) == std::future_status::ready) {
            auto response = observe_future.get();
            if (panel_alive_.load()) {
                if (response->success && response->point3d_base.size() >= 3) {
                    current_target_grasp_.clear();
                    for (float val : response->point3d_base) {
                        current_target_grasp_.append(static_cast<double>(val));
                    }
                }
                Q_EMIT observeResult(response->success,
                                     QString::fromStdString(response->category),
                                     response->score,
                                     QString::fromStdString(response->error_message));
            }
        } else {
            if (panel_alive_.load()) {
                Q_EMIT observeResult(false, "", 0, "Observe timeout");
            }
        }
    }).detach();
}

void PiperGraspPanel::updateAutoStatistics()
{
    double success_rate = 0;
    if (auto_cycle_count_ > 0) {
        success_rate = (double)auto_success_count_ / auto_cycle_count_ * 100;
    }

    QTime elapsed_time = QTime(0, 0).addSecs(auto_start_time_.secsTo(QTime::currentTime()));

    int avg_secs = 0;
    if (auto_cycle_count_ > 0) {
        avg_secs = auto_start_time_.secsTo(QTime::currentTime()) / auto_cycle_count_;
    }

    label_cycle_count_->setText(QString("Cycles: %1").arg(auto_cycle_count_));
    label_success_count_->setText(QString("Success: %1").arg(auto_success_count_));
    label_fail_count_->setText(QString("Failed: %1").arg(auto_fail_count_));
    label_success_rate_->setText(QString("Rate: %1%").arg(success_rate, 0, 'f', 1));

    if (auto_object_counts_.isEmpty()) {
        label_object_types_->setText("Objects: -");
    } else {
        QStringList object_list;
        for (auto it = auto_object_counts_.begin(); it != auto_object_counts_.end(); ++it) {
            object_list << QString("%1(%2)").arg(it.key()).arg(it.value());
        }
        label_object_types_->setText(QString("Objects: %1").arg(object_list.join(", ")));
    }

    label_duration_->setText(QString("Duration: %1").arg(elapsed_time.toString("mm:ss")));
    label_avg_time_->setText(QString("Avg: %1s/cycle").arg(avg_secs));
}

void PiperGraspPanel::resetAutoStatistics()
{
    auto_cycle_count_ = 0;
    auto_success_count_ = 0;
    auto_fail_count_ = 0;
    auto_consecutive_failures_ = 0;
    auto_consecutive_empty_observes_ = 0;
    auto_object_counts_.clear();
    auto_start_time_ = QTime::currentTime();

    updateAutoStatistics();
}

void PiperGraspPanel::checkAutoTerminationConditions()
{
    if (auto_consecutive_failures_ >= AUTO_MAX_CONSECUTIVE_FAILURES) {
        auto_clear_mode_.store(false);
        lets_grasp_state_.store(LetsGraspState::IDLE);
        updateLetsGraspUI();

        btn_auto_clear_->setEnabled(true);
        btn_end_clear_->setEnabled(false);
        btn_lets_grasp_->setEnabled(true);

        QString msg = QString("Auto Clear stopped: %1 consecutive failures")
                      .arg(AUTO_MAX_CONSECUTIVE_FAILURES);
        setLastStatus(msg, false);
        showErrorMessage(QString("[AUTO CLEAR STOPPED]\n%1").arg(msg));
        return;
    }

    if (auto_consecutive_empty_observes_ >= AUTO_MAX_CONSECUTIVE_EMPTY) {
        auto_clear_mode_.store(false);
        lets_grasp_state_.store(LetsGraspState::IDLE);
        updateLetsGraspUI();

        btn_auto_clear_->setEnabled(true);
        btn_end_clear_->setEnabled(false);
        btn_lets_grasp_->setEnabled(true);

        QString msg = QString("Auto Clear completed: No objects found (%1 attempts)")
                      .arg(AUTO_MAX_CONSECUTIVE_EMPTY);
        setLastStatus(msg, true);

        label_error_display_->setText(QString("[AUTO CLEAR COMPLETE]\nReason: No more objects found"));
        label_error_display_->setStyleSheet("font-family: monospace; color: green; background-color: #E0FFE0; padding: 5px; border: 2px solid green; font-weight: bold;");
        label_error_display_->setVisible(true);
        error_clear_timer_->start(ERROR_DISPLAY_DURATION_MS);
    }
}

// === Helpers ===

void PiperGraspPanel::updateConnectionDisplay(bool connected, bool enabled)
{
    QString text;
    QString style;

    if (connected && enabled) {
        text = "ENABLED";
        style = "color: green; font-weight: bold;";
    } else if (connected) {
        text = "CONNECTED";
        style = "color: orange; font-weight: bold;";
    } else {
        text = "DISCONNECTED";
        style = "color: red; font-weight: bold;";
    }

    label_connection_->setText(text);
    label_connection_->setStyleSheet(style);
}

void PiperGraspPanel::updateLetsGraspUI()
{
    QString status;
    int progress = 0;
    bool can_start = false;
    bool can_cancel = false;

    LetsGraspState state = lets_grasp_state_.load();
    switch (state) {
        case LetsGraspState::IDLE:
            status = "IDLE";
            progress = 0;
            can_start = true;
            label_stage_info_->setText("");
            break;
        case LetsGraspState::OBSERVING:
            status = "Observing...";
            progress = 15;
            can_cancel = true;
            break;
        case LetsGraspState::OBSERVE_ONLY:
            status = "Observe Only...";
            progress = 50;
            can_cancel = false;
            break;
        case LetsGraspState::PICKING:
            status = "Picking...";
            progress = 50;
            can_cancel = true;
            break;
        case LetsGraspState::PLACING:
            status = "Placing...";
            progress = 80;
            can_cancel = true;
            break;
        case LetsGraspState::CANCELLING:
            status = "Cancelling...";
            progress = bar_progress_->value();
            break;
        case LetsGraspState::COMPLETE:
            status = "Complete!";
            progress = 100;
            label_grasp_status_->setStyleSheet("color: green; font-weight: bold;");
            if (error_clear_timer_->isActive()) {
                error_clear_timer_->stop();
            }
            label_error_display_->setVisible(false);
            break;
        case LetsGraspState::ERROR:
            status = "Error";
            progress = 0;
            label_grasp_status_->setStyleSheet("color: red; font-weight: bold;");
            break;
        case LetsGraspState::AUTO_OBSERVING:
            status = QString("Auto Clear: Observing [%1]").arg(auto_cycle_count_);
            progress = 15;
            can_cancel = false;
            break;
        case LetsGraspState::AUTO_PICKING:
            status = QString("Auto Clear: Picking [%1]").arg(auto_cycle_count_);
            progress = 50;
            can_cancel = false;
            break;
        case LetsGraspState::AUTO_PLACING:
            status = QString("Auto Clear: Placing [%1]").arg(auto_cycle_count_);
            progress = 80;
            can_cancel = false;
            break;
        case LetsGraspState::AUTO_WAITING:
            status = "Auto Clear: Waiting next cycle...";
            progress = 100;
            can_cancel = false;
            break;
        case LetsGraspState::AUTO_FINISHING:
            status = "Auto Clear: Finishing...";
            progress = 100;
            can_cancel = false;
            break;
    }

    if (state != LetsGraspState::COMPLETE && state != LetsGraspState::ERROR) {
        label_grasp_status_->setStyleSheet("");
    }

    label_grasp_status_->setText(status);
    bar_progress_->setValue(progress);

    if (!auto_clear_mode_.load()) {
        btn_lets_grasp_->setEnabled(can_start);
        btn_cancel_->setEnabled(can_cancel);
        btn_observe_->setEnabled(can_start);
    } else {
        btn_observe_->setEnabled(false);
    }
}

void PiperGraspPanel::setLastStatus(const QString& msg, bool success)
{
    label_last_status_->setText(msg);
    label_last_status_->setStyleSheet(success ? "color: green;" : "color: red;");
}

void PiperGraspPanel::showErrorMessage(const QString& errorInfo)
{
    if (error_clear_timer_->isActive()) {
        error_clear_timer_->stop();
    }

    label_error_display_->setText(errorInfo);
    label_error_display_->setStyleSheet("font-family: monospace; color: red; background-color: #FFE0E0; padding: 5px; border: 2px solid red; font-weight: bold;");
    label_error_display_->setVisible(true);

    error_clear_timer_->start(ERROR_DISPLAY_DURATION_MS);
}

QString PiperGraspPanel::getModeText(uint8_t mode)
{
    switch (mode) {
        case 0: return "STANDBY";
        case 1: return "CAN";
        case 2: return "WIFI";
        case 3: return "ETHERNET";
        default: return QString("MODE_%1").arg(mode);
    }
}

QString PiperGraspPanel::getMotionText(uint8_t status)
{
    switch (status) {
        case 0: return "IDLE";
        case 1: return "MOVING";
        case 2: return "REACHED";
        default: return QString("S_%1").arg(status);
    }
}

// === Save/Load ===

void PiperGraspPanel::save(rviz_common::Config config) const
{
    rviz_common::Panel::save(config);
    config.mapSetValue("prompt", current_prompt_);
}

void PiperGraspPanel::load(const rviz_common::Config& config)
{
    rviz_common::Panel::load(config);
    QString prompt;
    if (config.mapGetString("prompt", &prompt)) {
        current_prompt_ = prompt;
        edit_prompt_->setText(prompt);
    }
}

}  // namespace piper_grasp

PLUGINLIB_EXPORT_CLASS(piper_grasp::PiperGraspPanel, rviz_common::Panel)
