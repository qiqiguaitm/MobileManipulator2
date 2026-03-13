/**
 * @file cleaner_panel.cpp
 * @brief CleanerPanel — main control panel for manual pick-and-place.
 *
 * State display + GO/CANCEL buttons.
 * Object selection is handled by ObjectSelectorNode (3D markers) or PerceptionPanel.
 */

#include "cleaner_manager/cleaner_panel.h"
#include <pluginlib/class_list_macros.hpp>
#include <chrono>
#include <cmath>

namespace cleaner_manager
{

// State → color mapping
QString CleanerPanel::stateColor(uint8_t state)
{
    switch (state) {
        case 0: return "color: #888888;";        // IDLE
        case 1: return "color: #FF8800;";        // DETECTING
        case 2: return "color: #4499FF;";        // WAITING_SELECT
        case 3: return "color: #FFCC00;";        // WAITING_CONFIRM
        case 4: return "color: #00BB00;";        // NAVIGATING
        case 5: return "color: #00BB00;";        // APPROACHING
        case 6: return "color: #00AAFF;";        // OBSERVING
        case 7: return "color: #FF6600;";        // PICKING
        case 8: return "color: #FF6600;";        // PLACING
        case 9: return "color: #FF0000;";        // ERROR
        default: return "color: #888888;";
    }
}

// Autonomous mode state → color mapping
QString CleanerPanel::autoStateColor(uint8_t state)
{
    switch (state) {
        case 0: return "color: #888888;";        // IDLE
        case 1: return "color: #4499FF;";        // PLANNING
        case 2: return "color: #00BB00;";        // NAVIGATING
        case 3: return "color: #FF8800;";        // PICKING
        case 4: return "color: #FF0000;";        // ERROR
        case 5: return "color: #00BB00;";        // COMPLETED
        default: return "color: #888888;";
    }
}

CleanerPanel::CleanerPanel(QWidget* parent)
    : rviz_common::Panel(parent)
{
    spin_timer_ = new QTimer(this);
    connect(spin_timer_, &QTimer::timeout, this, &CleanerPanel::spinOnce);

    setupUi();

    connect(this, &CleanerPanel::statusUpdated,
            this, &CleanerPanel::handleStatusUpdated, Qt::QueuedConnection);
    connect(this, &CleanerPanel::chassisUpdated,
            this, &CleanerPanel::handleChassisUpdated, Qt::QueuedConnection);
    connect(this, &CleanerPanel::armUpdated,
            this, &CleanerPanel::handleArmUpdated, Qt::QueuedConnection);
    connect(this, &CleanerPanel::cleanerManagerStatusUpdated,
            this, &CleanerPanel::handleCleanerManagerStatusUpdated, Qt::QueuedConnection);
}

CleanerPanel::~CleanerPanel()
{
    panel_alive_.store(false);
    if (spin_timer_) spin_timer_->stop();
    if (executor_) {
        try { executor_->cancel(); } catch (...) {}
    }
}

void CleanerPanel::onInitialize()
{
    auto context = getDisplayContext();
    if (!context) return;

    try {
        auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        std::string node_name = "cleaner_panel_" + std::to_string(now % 100000);
        node_ = std::make_shared<rclcpp::Node>(node_name);
        executor_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
        executor_->add_node(node_);

        setupRos();
        spin_timer_->start(100);  // 10Hz

        // 延迟2s发布初始抓取过滤配置 (等待 DDS discovery + load() 恢复保存值)
        QTimer::singleShot(2000, this, [this]() { publishGraspConfig(); });

        RCLCPP_INFO(node_->get_logger(), "[CleanerPanel] initialized");
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("cleaner_panel"),
                     "Init failed: %s", e.what());
    }
}

void CleanerPanel::save(rviz_common::Config config) const
{
    rviz_common::Panel::save(config);
    config.mapSetValue("grasp_z_min",    edit_grasp_z_min_->text());
    config.mapSetValue("grasp_z_max",    edit_grasp_z_max_->text());
    config.mapSetValue("grasp_dist_max", edit_grasp_dist_max_->text());
    config.mapSetValue("grasp_size_min", edit_grasp_size_min_->text());
    config.mapSetValue("grasp_size_max", edit_grasp_size_max_->text());
}

void CleanerPanel::load(const rviz_common::Config& config)
{
    rviz_common::Panel::load(config);
    QString v;
    if (config.mapGetString("grasp_z_min",    &v)) edit_grasp_z_min_->setText(v);
    if (config.mapGetString("grasp_z_max",    &v)) edit_grasp_z_max_->setText(v);
    if (config.mapGetString("grasp_dist_max", &v)) edit_grasp_dist_max_->setText(v);
    if (config.mapGetString("grasp_size_min", &v)) edit_grasp_size_min_->setText(v);
    if (config.mapGetString("grasp_size_max", &v)) edit_grasp_size_max_->setText(v);
}

void CleanerPanel::setupUi()
{
    QVBoxLayout* outer = new QVBoxLayout(this);
    outer->setSpacing(0);
    outer->setContentsMargins(0, 0, 0, 0);

    QScrollArea* scroll = new QScrollArea();
    scroll->setWidgetResizable(true);
    scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    scroll->setFrameShape(QFrame::NoFrame);

    QWidget* content = new QWidget();
    QVBoxLayout* layout = new QVBoxLayout(content);
    layout->setSpacing(4);
    layout->setContentsMargins(6, 6, 6, 6);

    // === Title ===
    QLabel* title = new QLabel("<b>CLEANER CONTROL</b>");
    title->setAlignment(Qt::AlignCenter);
    layout->addWidget(title);

    // === 状态 ===
    QGroupBox* grp_state = new QGroupBox("状态");
    QVBoxLayout* state_layout = new QVBoxLayout(grp_state);
    state_layout->setSpacing(2);

    lbl_state_ = new QLabel("● IDLE");
    lbl_state_->setStyleSheet("color: #888888; font-weight: bold; font-size: 13px;");
    state_layout->addWidget(lbl_state_);

    lbl_message_ = new QLabel("");
    lbl_message_->setStyleSheet("color: gray; font-size: 11px;");
    lbl_message_->setWordWrap(true);
    state_layout->addWidget(lbl_message_);

    layout->addWidget(grp_state);

    // === 底盘 ===
    QGroupBox* grp_chassis = new QGroupBox("底盘");
    QVBoxLayout* chassis_layout = new QVBoxLayout(grp_chassis);
    chassis_layout->setSpacing(2);

    lbl_chassis_status_ = new QLabel("● 未知");
    lbl_chassis_status_->setStyleSheet("color: gray;");
    chassis_layout->addWidget(lbl_chassis_status_);

    lbl_battery_ = new QLabel("电压: --");
    chassis_layout->addWidget(lbl_battery_);

    lbl_chassis_vel_ = new QLabel("速度: -- m/s  -- rad/s");
    chassis_layout->addWidget(lbl_chassis_vel_);

    layout->addWidget(grp_chassis);

    // === 机械臂 ===
    QGroupBox* grp_arm = new QGroupBox("机械臂");
    QVBoxLayout* arm_layout = new QVBoxLayout(grp_arm);
    arm_layout->setSpacing(2);

    lbl_arm_status_ = new QLabel("● 未知");
    lbl_arm_status_->setStyleSheet("color: gray;");
    arm_layout->addWidget(lbl_arm_status_);

    layout->addWidget(grp_arm);

    // === 目标池 ===
    QGroupBox* grp_target = new QGroupBox("目标池");
    QVBoxLayout* target_layout = new QVBoxLayout(grp_target);
    target_layout->setSpacing(2);

    lbl_current_target_ = new QLabel("当前: --");
    lbl_current_target_->setStyleSheet("color: #00AA00; font-size: 11px;");
    target_layout->addWidget(lbl_current_target_);

    list_targets_ = new QListWidget();
    list_targets_->setAlternatingRowColors(true);
    list_targets_->setMinimumHeight(100);
    list_targets_->setStyleSheet(
        "QListWidget { font-family: monospace; font-size: 11px; }"
        "QListWidget::item:selected { background-color: #2E5A88; color: white; }");
    target_layout->addWidget(list_targets_);

    btn_reset_pool_ = new QPushButton("清空目标池");
    btn_reset_pool_->setStyleSheet(
        "QPushButton { background-color: #555500; color: white; }"
        "QPushButton:hover { background-color: #777700; }");
    connect(btn_reset_pool_, &QPushButton::clicked, this, &CleanerPanel::onResetPoolClicked);
    target_layout->addWidget(btn_reset_pool_);

    layout->addWidget(grp_target);

    // === 进度 (PICKING/PLACING 时显示) ===
    grp_progress_ = new QGroupBox("进度");
    QVBoxLayout* prog_layout = new QVBoxLayout(grp_progress_);
    prog_layout->setSpacing(2);

    QHBoxLayout* prog_row = new QHBoxLayout();
    bar_progress_ = new QProgressBar();
    bar_progress_->setRange(0, 100);
    bar_progress_->setValue(0);
    prog_row->addWidget(bar_progress_);

    lbl_grasp_step_ = new QLabel("--");
    lbl_grasp_step_->setMinimumWidth(60);
    prog_row->addWidget(lbl_grasp_step_);
    prog_layout->addLayout(prog_row);

    grp_progress_->setVisible(false);
    layout->addWidget(grp_progress_);

    // === 抓取过滤 ===
    QGroupBox* grp_grasp = new QGroupBox("抓取过滤");
    QVBoxLayout* grasp_layout = new QVBoxLayout(grp_grasp);
    grasp_layout->setSpacing(2);

    QHBoxLayout* z_row = new QHBoxLayout();
    z_row->addWidget(new QLabel("高度 z:"));
    edit_grasp_z_min_ = new QLineEdit("-0.30");
    edit_grasp_z_min_->setMaximumWidth(50);
    z_row->addWidget(edit_grasp_z_min_);
    z_row->addWidget(new QLabel("~"));
    edit_grasp_z_max_ = new QLineEdit("0.50");
    edit_grasp_z_max_->setMaximumWidth(50);
    z_row->addWidget(edit_grasp_z_max_);
    z_row->addWidget(new QLabel("m"));
    z_row->addStretch();
    grasp_layout->addLayout(z_row);

    QHBoxLayout* size_row = new QHBoxLayout();
    size_row->addWidget(new QLabel("尺寸:"));
    edit_grasp_size_min_ = new QLineEdit("0.02");
    edit_grasp_size_min_->setMaximumWidth(50);
    size_row->addWidget(edit_grasp_size_min_);
    size_row->addWidget(new QLabel("~"));
    edit_grasp_size_max_ = new QLineEdit("0.25");
    edit_grasp_size_max_->setMaximumWidth(50);
    size_row->addWidget(edit_grasp_size_max_);
    size_row->addWidget(new QLabel("m"));
    size_row->addStretch();
    grasp_layout->addLayout(size_row);

    QHBoxLayout* dist_row = new QHBoxLayout();
    dist_row->addWidget(new QLabel("最远距离:"));
    edit_grasp_dist_max_ = new QLineEdit("6.00");
    edit_grasp_dist_max_->setMaximumWidth(55);
    dist_row->addWidget(edit_grasp_dist_max_);
    dist_row->addWidget(new QLabel("m"));
    dist_row->addStretch();
    grasp_layout->addLayout(dist_row);

    btn_apply_grasp_ = new QPushButton("应用");
    btn_apply_grasp_->setStyleSheet(
        "QPushButton { background-color: #336699; color: white; }"
        "QPushButton:hover { background-color: #4488AA; }");
    connect(btn_apply_grasp_, &QPushButton::clicked, this, &CleanerPanel::onApplyGraspClicked);
    grasp_layout->addWidget(btn_apply_grasp_);

    layout->addWidget(grp_grasp);

    // === 自主模式 ===
    grp_auto_ = new QGroupBox("自主模式");
    QVBoxLayout* auto_layout = new QVBoxLayout(grp_auto_);
    auto_layout->setSpacing(2);

    lbl_auto_state_ = new QLabel("● IDLE");
    lbl_auto_state_->setStyleSheet("color: #888888; font-weight: bold;");
    auto_layout->addWidget(lbl_auto_state_);

    lbl_auto_stats_ = new QLabel("目标: 总0 拾0 失0 剩0");
    lbl_auto_stats_->setStyleSheet("color: gray; font-size: 11px;");
    auto_layout->addWidget(lbl_auto_stats_);

    lbl_auto_error_ = new QLabel("");
    lbl_auto_error_->setStyleSheet("color: #FF4400; font-size: 11px;");
    lbl_auto_error_->setWordWrap(true);
    lbl_auto_error_->setVisible(false);
    auto_layout->addWidget(lbl_auto_error_);

    QHBoxLayout* auto_btn_row = new QHBoxLayout();
    btn_start_auto_ = new QPushButton("START AUTO");
    btn_start_auto_->setMinimumHeight(36);
    btn_start_auto_->setStyleSheet(
        "QPushButton { background-color: #005588; color: white; font-weight: bold; }"
        "QPushButton:hover { background-color: #0077AA; }"
        "QPushButton:disabled { background-color: #555555; color: #999999; }");
    connect(btn_start_auto_, &QPushButton::clicked, this, &CleanerPanel::onStartAutoClicked);
    auto_btn_row->addWidget(btn_start_auto_);

    btn_stop_auto_ = new QPushButton("STOP AUTO");
    btn_stop_auto_->setMinimumHeight(36);
    btn_stop_auto_->setEnabled(false);
    btn_stop_auto_->setStyleSheet(
        "QPushButton { background-color: #772200; color: white; font-weight: bold; }"
        "QPushButton:hover { background-color: #993300; }"
        "QPushButton:disabled { background-color: #555555; color: #999999; }");
    connect(btn_stop_auto_, &QPushButton::clicked, this, &CleanerPanel::onStopAutoClicked);
    auto_btn_row->addWidget(btn_stop_auto_);

    auto_layout->addLayout(auto_btn_row);
    layout->addWidget(grp_auto_);

    // === 按钮 ===
    QHBoxLayout* btn_row = new QHBoxLayout();

    btn_go_ = new QPushButton("GO");
    btn_go_->setMinimumHeight(44);
    btn_go_->setEnabled(false);
    btn_go_->setStyleSheet(
        "QPushButton { background-color: #007700; color: white; font-weight: bold; font-size: 16px; }"
        "QPushButton:hover { background-color: #009900; }"
        "QPushButton:disabled { background-color: #555555; color: #999999; }");
    connect(btn_go_, &QPushButton::clicked, this, &CleanerPanel::onGoClicked);
    btn_row->addWidget(btn_go_);

    btn_cancel_ = new QPushButton("CANCEL");
    btn_cancel_->setMinimumHeight(44);
    btn_cancel_->setEnabled(false);
    btn_cancel_->setStyleSheet(
        "QPushButton { background-color: #770000; color: white; font-weight: bold; font-size: 14px; }"
        "QPushButton:hover { background-color: #990000; }"
        "QPushButton:disabled { background-color: #555555; color: #999999; }");
    connect(btn_cancel_, &QPushButton::clicked, this, &CleanerPanel::onCancelClicked);
    btn_row->addWidget(btn_cancel_);

    layout->addLayout(btn_row);
    layout->addStretch();

    scroll->setWidget(content);
    outer->addWidget(scroll);
}

void CleanerPanel::setupRos()
{
    sub_status_ = node_->create_subscription<cleaner_manager::msg::ManualStatus>(
        "/manual_control_node/status", 10,
        [this](const cleaner_manager::msg::ManualStatus::SharedPtr msg) {
            statusCb(msg);
        });

    sub_chassis_ = node_->create_subscription<tracer_msgs::msg::TracerStatus>(
        "/tracer_status", 10,
        [this](const tracer_msgs::msg::TracerStatus::SharedPtr msg) {
            chassisCb(msg);
        });

    sub_arm_ = node_->create_subscription<piper_msgs::msg::PiperStatus>(
        "/piper/status", 10,
        [this](const piper_msgs::msg::PiperStatus::SharedPtr msg) {
            armCb(msg);
        });

    cmd_cli_ = node_->create_client<cleaner_manager::srv::ManualCommand>(
        "/manual_control_node/command");

    sub_auto_status_ = node_->create_subscription<cleaner_manager::msg::CleanerManagerStatus>(
        "/cleaner_manager_node/status", 10,
        [this](const cleaner_manager::msg::CleanerManagerStatus::SharedPtr msg) {
            cleanerManagerStatusCb(msg);
        });

    start_auto_cli_ = node_->create_client<std_srvs::srv::Trigger>(
        "/cleaner_manager_node/start");
    stop_auto_cli_ = node_->create_client<std_srvs::srv::Trigger>(
        "/cleaner_manager_node/abort");
    reset_pool_cli_ = node_->create_client<std_srvs::srv::Trigger>(
        "/cleaner_manager_node/reset_pool");

    pub_grasp_config_ = node_->create_publisher<perception::msg::PerceptionConfig>(
        "/perception/config", 10);
}

void CleanerPanel::statusCb(const cleaner_manager::msg::ManualStatus::SharedPtr msg)
{
    if (!panel_alive_.load()) return;
    emit statusUpdated(
        msg->state,
        QString::fromStdString(msg->state_name),
        QString::fromStdString(msg->message),
        QString::fromStdString(msg->target_category),
        QString::fromStdString(msg->target_object_id),
        msg->target_distance,
        msg->target_position.x, msg->target_position.y,
        msg->robot_position.x, msg->robot_position.y,
        QString::fromStdString(msg->grasp_step),
        static_cast<double>(msg->grasp_progress));
}

void CleanerPanel::chassisCb(const tracer_msgs::msg::TracerStatus::SharedPtr msg)
{
    if (!panel_alive_.load()) return;
    emit chassisUpdated(
        msg->linear_velocity,
        msg->angular_velocity,
        msg->battery_voltage,
        msg->control_mode,
        msg->error_code);
}

void CleanerPanel::armCb(const piper_msgs::msg::PiperStatus::SharedPtr msg)
{
    if (!panel_alive_.load()) return;
    emit armUpdated(
        msg->connected,
        msg->enabled,
        QString::fromStdString(msg->error_state));
}

void CleanerPanel::cleanerManagerStatusCb(
    const cleaner_manager::msg::CleanerManagerStatus::SharedPtr msg)
{
    if (!panel_alive_.load()) return;

    std::vector<TargetInfo> targets;
    targets.reserve(msg->pool_targets.size());
    for (const auto& t : msg->pool_targets) {
        TargetInfo info;
        info.category = t.category;
        info.pos_x = t.pos_x;
        info.pos_y = t.pos_y;
        info.score = t.score;
        info.status = t.status;
        info.observations = t.observations;
        info.attempts = t.attempts;
        info.last_seen_age = t.last_seen_age;
        info.viable = t.viable;
        targets.push_back(std::move(info));
    }
    {
        std::lock_guard<std::mutex> lock(targets_mutex_);
        cached_targets_ = std::move(targets);
    }
    targets_dirty_.store(true);

    emit cleanerManagerStatusUpdated(
        msg->state,
        QString::fromStdString(msg->state_name),
        msg->targets_total,
        msg->targets_picked,
        msg->targets_failed,
        msg->targets_remaining,
        QString::fromStdString(msg->error_message),
        QString::fromStdString(msg->current_target_category));
}

void CleanerPanel::handleCleanerManagerStatusUpdated(
    uint8_t state,
    const QString& state_name,
    uint32_t targets_total,
    uint32_t targets_picked,
    uint32_t targets_failed,
    uint32_t targets_remaining,
    const QString& error_message,
    const QString& current_target_category)
{
    auto_state_ = state;

    lbl_auto_state_->setText(QString("● %1").arg(state_name));
    lbl_auto_state_->setStyleSheet(autoStateColor(state) + " font-weight: bold;");

    lbl_auto_stats_->setText(QString("目标: 总%1 拾%2 失%3 剩%4")
        .arg(targets_total).arg(targets_picked).arg(targets_failed).arg(targets_remaining));

    // Update current target from auto mode (overrides manual mode display)
    if (!current_target_category.isEmpty()) {
        lbl_current_target_->setText(QString("当前(auto): %1").arg(current_target_category));
        lbl_current_target_->setStyleSheet("color: #00AA00; font-size: 11px;");
    } else if (state == 0) {
        // Auto IDLE: clear auto target display
        lbl_current_target_->setText("当前: --");
        lbl_current_target_->setStyleSheet("color: gray; font-size: 11px;");
    }

    bool has_error = (state == 4 && !error_message.isEmpty());  // PickState::ERROR = 4
    lbl_auto_error_->setVisible(has_error);
    if (has_error) lbl_auto_error_->setText(error_message);

    // AUTO is "running" for states PLANNING(1)..PICKING(3)
    bool auto_running = (state >= 1 && state <= 3);
    btn_start_auto_->setEnabled(!auto_running);
    btn_stop_auto_->setEnabled(auto_running);

    // Disable manual GO when auto is running
    if (auto_running) {
        btn_go_->setEnabled(false);
    } else {
        // Restore manual GO state based on current manual state
        btn_go_->setEnabled(current_state_ == 3);
    }
}

void CleanerPanel::handleStatusUpdated(
    uint8_t state,
    const QString& state_name,
    const QString& message,
    const QString& target_category,
    const QString& target_object_id,
    double target_distance,
    double target_x, double target_y,
    double /*robot_x*/, double /*robot_y*/,
    const QString& grasp_step,
    double grasp_progress)
{
    current_state_ = state;

    lbl_state_->setText(QString("● %1").arg(state_name));
    lbl_state_->setStyleSheet(stateColor(state) + " font-weight: bold; font-size: 13px;");
    lbl_message_->setText(message);

    // Current target (active pick target)
    if (!target_category.isEmpty()) {
        lbl_current_target_->setText(
            QString("当前: %1 (%2)  %3m  (%4, %5)")
                .arg(target_category)
                .arg(target_object_id)
                .arg(target_distance, 0, 'f', 2)
                .arg(target_x, 0, 'f', 1)
                .arg(target_y, 0, 'f', 1));
        lbl_current_target_->setStyleSheet("color: #00AA00; font-size: 11px;");
    } else {
        lbl_current_target_->setText("当前: --");
        lbl_current_target_->setStyleSheet("color: gray; font-size: 11px;");
    }

    // Progress (state 7=PICKING, 8=PLACING)
    bool show_progress = (state == 7 || state == 8);
    grp_progress_->setVisible(show_progress);
    if (show_progress) {
        bar_progress_->setValue(static_cast<int>(grasp_progress * 100.0));
        lbl_grasp_step_->setText(grasp_step.isEmpty() ? "--" : grasp_step);
    }

    // Button enable logic
    // GO: only in WAITING_CONFIRM (state 3), disabled when auto mode is running
    bool auto_running = (auto_state_ >= 1 && auto_state_ <= 3);
    btn_go_->setEnabled(state == 3 && !auto_running);
    // CANCEL: any active state (not IDLE=0, not ERROR=9)
    btn_cancel_->setEnabled(state != 0 && state != 9);
}

void CleanerPanel::handleChassisUpdated(
    double linear_vel,
    double angular_vel,
    double battery_voltage,
    uint8_t /*control_mode*/,
    uint16_t error_code)
{
    QString status_style = (error_code == 0)
        ? "color: #00AA00;" : "color: #FF4400;";
    QString status_text = (error_code == 0) ? "● 正常" : QString("● 错误 0x%1").arg(error_code, 0, 16);
    lbl_chassis_status_->setText(status_text);
    lbl_chassis_status_->setStyleSheet(status_style);

    // Battery: 24V nominal, 29.4V max, 20V min
    double pct = (battery_voltage - 20.0) / (29.4 - 20.0) * 100.0;
    pct = std::max(0.0, std::min(100.0, pct));
    lbl_battery_->setText(
        QString("电压: %1V  [%2%]").arg(battery_voltage, 0, 'f', 1).arg(static_cast<int>(pct)));

    lbl_chassis_vel_->setText(
        QString("速度: %1 m/s  %2 rad/s")
        .arg(linear_vel, 0, 'f', 2)
        .arg(angular_vel, 0, 'f', 2));
}

void CleanerPanel::handleArmUpdated(
    bool connected,
    bool enabled,
    const QString& error_state)
{
    if (!connected) {
        lbl_arm_status_->setText("● 未连接");
        lbl_arm_status_->setStyleSheet("color: red;");
    } else if (!enabled) {
        lbl_arm_status_->setText("● 未使能");
        lbl_arm_status_->setStyleSheet("color: orange;");
    } else if (error_state == "NORMAL") {
        lbl_arm_status_->setText("● 已使能 正常");
        lbl_arm_status_->setStyleSheet("color: #00AA00;");
    } else {
        lbl_arm_status_->setText(QString("● %1").arg(error_state));
        lbl_arm_status_->setStyleSheet("color: #FF4400;");
    }
}

void CleanerPanel::onGoClicked()
{
    callCommand(cleaner_manager::srv::ManualCommand::Request::CMD_GO);
}

void CleanerPanel::onCancelClicked()
{
    callCommand(cleaner_manager::srv::ManualCommand::Request::CMD_CANCEL);
}

void CleanerPanel::onStartAutoClicked()
{
    if (!start_auto_cli_ || !node_) return;
    auto req = std::make_shared<std_srvs::srv::Trigger::Request>();
    start_auto_cli_->async_send_request(req);
}

void CleanerPanel::onStopAutoClicked()
{
    if (!stop_auto_cli_ || !node_) return;
    auto req = std::make_shared<std_srvs::srv::Trigger::Request>();
    stop_auto_cli_->async_send_request(req);
}

void CleanerPanel::onResetPoolClicked()
{
    if (!reset_pool_cli_ || !node_) return;
    auto req = std::make_shared<std_srvs::srv::Trigger::Request>();
    reset_pool_cli_->async_send_request(req);
}

void CleanerPanel::onApplyGraspClicked()
{
    publishGraspConfig();
}

void CleanerPanel::publishGraspConfig()
{
    if (!pub_grasp_config_ || !node_) return;
    auto msg = std::make_shared<perception::msg::PerceptionConfig>();
    msg->update_grasp_filter = true;
    try { msg->grasp_z_min            = std::stof(edit_grasp_z_min_->text().toStdString()); } catch (...) { msg->grasp_z_min = -0.3f; }
    try { msg->grasp_z_max            = std::stof(edit_grasp_z_max_->text().toStdString()); } catch (...) { msg->grasp_z_max =  0.5f; }
    try { msg->grasp_distance_max     = std::stof(edit_grasp_dist_max_->text().toStdString()); } catch (...) { msg->grasp_distance_max = 6.0f; }
    try { msg->grasp_physical_min_size = std::stof(edit_grasp_size_min_->text().toStdString()); } catch (...) { msg->grasp_physical_min_size = 0.02f; }
    try { msg->grasp_physical_max_size = std::stof(edit_grasp_size_max_->text().toStdString()); } catch (...) { msg->grasp_physical_max_size = 0.25f; }
    pub_grasp_config_->publish(*msg);
    RCLCPP_INFO(node_->get_logger(),
        "[CleanerPanel] grasp filter: z=[%.2f,%.2f]m size=[%.2f,%.2f]m dist=%.1fm",
        msg->grasp_z_min, msg->grasp_z_max,
        msg->grasp_physical_min_size, msg->grasp_physical_max_size,
        msg->grasp_distance_max);
}

void CleanerPanel::callCommand(uint8_t cmd)
{
    if (!cmd_cli_ || !node_) return;
    auto req = std::make_shared<cleaner_manager::srv::ManualCommand::Request>();
    req->command = cmd;
    cmd_cli_->async_send_request(req);
}

void CleanerPanel::spinOnce()
{
    if (executor_) {
        executor_->spin_some(std::chrono::milliseconds(10));
    }

    if (targets_dirty_.exchange(false))
        updateTargetListUI();
}

void CleanerPanel::updateTargetListUI()
{
    std::vector<TargetInfo> snap;
    {
        std::lock_guard<std::mutex> lock(targets_mutex_);
        snap = cached_targets_;
    }

    list_targets_->clear();
    for (const auto& t : snap) {
        // Only show viable ACTIVE targets; stale/low-obs and PICKED/FAILED are noise
        if (t.status != 0 || !t.viable) continue;

        QString status_str = t.viable ? "OK" : "stale";
        // viable: default palette color; stale: dim gray
        QColor fg = t.viable
            ? list_targets_->palette().color(QPalette::Text)
            : QColor("#999999");

        QString text = QString("%1 (%2,%3) s=%4 obs=%5 %6")
            .arg(QString::fromStdString(t.category), -8)
            .arg(t.pos_x, 5, 'f', 1)
            .arg(t.pos_y, 5, 'f', 1)
            .arg(t.score, 0, 'f', 2)
            .arg(t.observations)
            .arg(status_str);
        auto* item = new QListWidgetItem(text);
        item->setForeground(fg);
        list_targets_->addItem(item);
    }
}

}  // namespace cleaner_manager

PLUGINLIB_EXPORT_CLASS(cleaner_manager::CleanerPanel, rviz_common::Panel)
