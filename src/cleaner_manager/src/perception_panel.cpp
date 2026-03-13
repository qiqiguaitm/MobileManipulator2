/**
 * @file perception_panel.cpp
 * @brief PerceptionPanel — real-time object list + perception config.
 *
 * Threading model:
 *   ROS callbacks (executor thread) → update cached_objects_ + set objects_dirty_
 *   spinOnce() (Qt timer, main thread) → if dirty, flush to list widget
 *   No std::vector in Qt signals → no qRegisterMetaType needed.
 */

#include "cleaner_manager/perception_panel.h"
#include <pluginlib/class_list_macros.hpp>
#include <chrono>

namespace cleaner_manager
{

PerceptionPanel::PerceptionPanel(QWidget* parent)
    : rviz_common::Panel(parent)
{
    spin_timer_ = new QTimer(this);
    connect(spin_timer_, &QTimer::timeout, this, &PerceptionPanel::spinOnce);

    setupUi();

    connect(this, &PerceptionPanel::perceptionStatusUpdated,
            this, &PerceptionPanel::handlePerceptionStatusUpdated, Qt::QueuedConnection);
}

PerceptionPanel::~PerceptionPanel()
{
    panel_alive_.store(false);
    if (spin_timer_) spin_timer_->stop();
    if (executor_) {
        try { executor_->cancel(); } catch (...) {}
    }
}

void PerceptionPanel::onInitialize()
{
    auto context = getDisplayContext();
    if (!context) return;

    try {
        auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        std::string node_name = "perception_panel_" + std::to_string(now % 100000);
        node_ = std::make_shared<rclcpp::Node>(node_name);
        executor_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
        executor_->add_node(node_);

        last_objects_time_ = node_->now();
        setupRos();
        spin_timer_->start(100);  // 10Hz

        // 不自动发布配置 — 参数由 launch/yaml 管理，panel 只做显示和手动覆盖

        RCLCPP_INFO(node_->get_logger(), "[PerceptionPanel] initialized");
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("perception_panel"),
                     "Init failed: %s", e.what());
    }
}

void PerceptionPanel::save(rviz_common::Config config) const
{
    rviz_common::Panel::save(config);
    // prompt/min_score 不持久化到 .rviz，由 launch/yaml 管理
}

void PerceptionPanel::load(const rviz_common::Config& config)
{
    rviz_common::Panel::load(config);
    // 不从 .rviz 恢复 prompt/min_score，启动后由 /perception/status 回填
}

void PerceptionPanel::setupUi()
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
    QLabel* title = new QLabel("<b>PERCEPTION</b>");
    title->setAlignment(Qt::AlignCenter);
    layout->addWidget(title);

    // === 感知状态 ===
    QGroupBox* grp_status = new QGroupBox("感知状态");
    QVBoxLayout* status_layout = new QVBoxLayout(grp_status);
    status_layout->setSpacing(2);

    lbl_perc_status_ = new QLabel("● 离线");
    lbl_perc_status_->setStyleSheet("color: gray; font-weight: bold;");
    status_layout->addWidget(lbl_perc_status_);

    lbl_detect_time_ = new QLabel("耗时: --");
    status_layout->addWidget(lbl_detect_time_);

    lbl_object_count_ = new QLabel("物体数: 0");
    status_layout->addWidget(lbl_object_count_);

    layout->addWidget(grp_status);

    // === 配置 ===
    QGroupBox* grp_config = new QGroupBox("配置");
    QVBoxLayout* config_layout = new QVBoxLayout(grp_config);
    config_layout->setSpacing(2);

    QHBoxLayout* prompt_row = new QHBoxLayout();
    prompt_row->addWidget(new QLabel("提示词:"));
    edit_prompt_ = new QLineEdit("bottle.cup.box.can.toy.pen.bag");
    edit_prompt_->setPlaceholderText("bottle.cup.box.can.toy.pen.bag");
    prompt_row->addWidget(edit_prompt_);
    config_layout->addLayout(prompt_row);

    QHBoxLayout* score_row = new QHBoxLayout();
    score_row->addWidget(new QLabel("最低分:"));
    edit_min_score_ = new QLineEdit("0.25");
    edit_min_score_->setMaximumWidth(60);
    score_row->addWidget(edit_min_score_);
    score_row->addStretch();

    btn_apply_config_ = new QPushButton("应用");
    btn_apply_config_->setStyleSheet(
        "QPushButton { background-color: #336699; color: white; }"
        "QPushButton:hover { background-color: #4488AA; }");
    connect(btn_apply_config_, &QPushButton::clicked,
            this, &PerceptionPanel::onApplyConfigClicked);
    score_row->addWidget(btn_apply_config_);
    config_layout->addLayout(score_row);

    // 输入框回车自动应用
    connect(edit_prompt_, &QLineEdit::returnPressed,
            this, &PerceptionPanel::onApplyConfigClicked);
    connect(edit_min_score_, &QLineEdit::returnPressed,
            this, &PerceptionPanel::onApplyConfigClicked);

    layout->addWidget(grp_config);

    // === 检测物体列表 ===
    QGroupBox* grp_objects = new QGroupBox("检测物体 (点击选择)");
    QVBoxLayout* objects_layout = new QVBoxLayout(grp_objects);
    objects_layout->setSpacing(2);

    list_objects_ = new QListWidget();
    list_objects_->setAlternatingRowColors(true);
    list_objects_->setMinimumHeight(120);
    list_objects_->setStyleSheet(
        "QListWidget { font-family: monospace; font-size: 11px; }"
        "QListWidget::item:hover { background-color: #3A6EA5; color: white; }"
        "QListWidget::item:selected { background-color: #2E5A88; color: white; }");
    connect(list_objects_, &QListWidget::itemClicked,
            this, &PerceptionPanel::onObjectItemClicked);
    objects_layout->addWidget(list_objects_);

    layout->addWidget(grp_objects);
    layout->addStretch();

    scroll->setWidget(content);
    outer->addWidget(scroll);
}

void PerceptionPanel::setupRos()
{
    sub_perc_status_ = node_->create_subscription<perception::msg::PerceptionStatus>(
        "/perception/status", 10,
        [this](const perception::msg::PerceptionStatus::SharedPtr msg) {
            perceptionStatusCb(msg);
        });

    sub_objects_ = node_->create_subscription<perception::msg::Object3DArray>(
        "/multi_camera_perception/fused/objects_3d", 10,
        [this](const perception::msg::Object3DArray::SharedPtr msg) {
            objectsCb(msg);
        });

    // Subscribe to default_prompt from perception (single source of truth, latched)
    sub_default_prompt_ = node_->create_subscription<std_msgs::msg::String>(
        "/piper/default_prompt", rclcpp::QoS(1).transient_local().reliable(),
        [this](const std_msgs::msg::String::SharedPtr msg) {
            if (!panel_alive_.load() || msg->data.empty()) return;
            QMetaObject::invokeMethod(this, [this, prompt = QString::fromStdString(msg->data)]() {
                edit_prompt_->setText(prompt);
            }, Qt::QueuedConnection);
        });

    pub_config_ = node_->create_publisher<perception::msg::PerceptionConfig>(
        "/perception/config", 10);

    cmd_cli_ = node_->create_client<cleaner_manager::srv::ManualCommand>(
        "/manual_control_node/command");
}

void PerceptionPanel::perceptionStatusCb(
    const perception::msg::PerceptionStatus::SharedPtr msg)
{
    if (!panel_alive_.load()) return;
    // Qt primitive types only — safe for QueuedConnection
    emit perceptionStatusUpdated(
        QString::fromStdString(msg->prompt),
        msg->min_score,
        msg->object_count,
        msg->last_detect_time);
}

void PerceptionPanel::objectsCb(
    const perception::msg::Object3DArray::SharedPtr msg)
{
    if (!panel_alive_.load()) return;

    std::vector<ObjectInfo> objs;
    objs.reserve(msg->objects.size());
    // RealSense D435 @ 1280x720 的典型焦距，用于估算物理尺寸
    constexpr double kFx = 920.0, kFy = 920.0;
    for (const auto& o : msg->objects) {
        ObjectInfo info;
        info.object_id  = o.object_id;
        info.category   = o.category;
        info.distance   = o.distance;
        info.score      = o.score;
        info.source     = o.source_camera;
        info.pos_x      = o.position.x;
        info.pos_y      = o.position.y;
        info.pos_z      = o.position.z;
        info.depth      = o.depth;
        if (o.bbox.size() >= 4) {
            info.bbox_w = o.bbox[2] - o.bbox[0];
            info.bbox_h = o.bbox[3] - o.bbox[1];
            if (o.depth > 0.1) {
                info.phys_w = info.bbox_w * o.depth / kFx;
                info.phys_h = info.bbox_h * o.depth / kFy;
            }
        }
        objs.push_back(std::move(info));
    }

    {
        std::lock_guard<std::mutex> lock(objects_mutex_);
        cached_objects_ = std::move(objs);
    }
    objects_dirty_.store(true);
    last_objects_time_ = node_->now();
    perc_online_.store(true);
}

// Called on Qt main thread by spinOnce() — safe to touch widgets
void PerceptionPanel::updateObjectListUI()
{
    std::vector<ObjectInfo> snap;
    {
        std::lock_guard<std::mutex> lock(objects_mutex_);
        snap = cached_objects_;
    }

    lbl_perc_status_->setText(QString("● 在线 (%1个)").arg(snap.size()));
    lbl_perc_status_->setStyleSheet("color: #00AA00; font-weight: bold;");

    list_objects_->clear();
    for (const auto& obj : snap) {
        // 物理尺寸字符串
        QString size_str = (obj.phys_w > 0)
            ? QString("%1x%2cm").arg(obj.phys_w * 100, 0, 'f', 0).arg(obj.phys_h * 100, 0, 'f', 0)
            : QString("--");
        // 从 object_id (e.g. "bottle_3") 提取尾部数字作为 ID 列
        QString oid = QString::fromStdString(obj.object_id);
        int sep = oid.lastIndexOf('_');
        QString id_str = (sep >= 0) ? "#" + oid.mid(sep + 1) : oid;

        QString text = QString("%1 %2 d=%3m z=%4m %5 s=%6")
            .arg(QString::fromStdString(obj.category), -8)
            .arg(id_str, -4)
            .arg(obj.distance, 4, 'f', 1)
            .arg(obj.pos_z, 4, 'f', 2)
            .arg(size_str, -10)
            .arg(obj.score, 0, 'f', 2);
        auto* item = new QListWidgetItem(text);
        item->setToolTip(QString::fromStdString(
            obj.object_id + "\n"
            "pos(base_link): (" +
            std::to_string(obj.pos_x).substr(0, 5) + ", " +
            std::to_string(obj.pos_y).substr(0, 5) + ", " +
            std::to_string(obj.pos_z).substr(0, 5) + ")\n"
            "depth=" + std::to_string(obj.depth).substr(0, 4) + "m  "
            "bbox=" + std::to_string((int)obj.bbox_w) + "x" + std::to_string((int)obj.bbox_h) + "px\n"
            "cam=" + obj.source));
        list_objects_->addItem(item);
    }
}

void PerceptionPanel::handlePerceptionStatusUpdated(
    const QString& prompt,
    float min_score,
    uint32_t object_count,
    float last_detect_time)
{
    lbl_detect_time_->setText(QString("耗时: %1s").arg(last_detect_time, 0, 'f', 2));
    lbl_object_count_->setText(QString("物体数: %1").arg(object_count));

    // 回填当前节点的值（不在编辑状态时更新）
    if (!edit_prompt_->hasFocus())
        edit_prompt_->setText(prompt);
    if (!edit_min_score_->hasFocus())
        edit_min_score_->setText(QString::number(min_score, 'f', 2));
}

void PerceptionPanel::publishConfig()
{
    if (!pub_config_ || !node_) return;

    auto msg = std::make_shared<perception::msg::PerceptionConfig>();
    msg->prompt = edit_prompt_->text().toStdString();
    try {
        msg->min_score = std::stof(edit_min_score_->text().toStdString());
    } catch (...) {
        msg->min_score = 0.25f;
    }
    msg->iou_threshold = 0.5f;
    pub_config_->publish(*msg);

    RCLCPP_INFO(node_->get_logger(),
                "[PerceptionPanel] config applied: prompt=%s min_score=%.2f",
                msg->prompt.c_str(), msg->min_score);
}

void PerceptionPanel::onApplyConfigClicked()
{
    publishConfig();
}

void PerceptionPanel::onObjectItemClicked(QListWidgetItem* item)
{
    int row = list_objects_->row(item);

    ObjectInfo obj;
    {
        std::lock_guard<std::mutex> lock(objects_mutex_);
        if (row < 0 || row >= static_cast<int>(cached_objects_.size())) return;
        obj = cached_objects_[row];
    }

    if (!cmd_cli_ || !node_) return;

    auto req = std::make_shared<cleaner_manager::srv::ManualCommand::Request>();
    req->command        = cleaner_manager::srv::ManualCommand::Request::CMD_SELECT_OBJECT;
    req->object_id      = obj.object_id;
    req->category       = obj.category;
    req->distance       = obj.distance;
    req->target_position.x = obj.pos_x;
    req->target_position.y = obj.pos_y;
    req->target_position.z = obj.pos_z;
    req->position_frame = "base_link";  // manual_control_node transforms to map

    cmd_cli_->async_send_request(req);

    RCLCPP_INFO(node_->get_logger(),
                "[PerceptionPanel] selected: %s (%s) %.1fm",
                obj.category.c_str(), obj.object_id.c_str(), obj.distance);
}

void PerceptionPanel::spinOnce()
{
    if (executor_)
        executor_->spin_some(std::chrono::milliseconds(10));

    // Flush object list to UI if new data arrived
    if (objects_dirty_.exchange(false))
        updateObjectListUI();

    // Perception timeout: 5s without new objects → show offline
    if (node_ && perc_online_.load()) {
        auto age = (node_->now() - last_objects_time_).seconds();
        if (age > 5.0) {
            perc_online_.store(false);
            lbl_perc_status_->setText("● 离线 (超时)");
            lbl_perc_status_->setStyleSheet("color: gray; font-weight: bold;");
        }
    }
}

}  // namespace cleaner_manager

PLUGINLIB_EXPORT_CLASS(cleaner_manager::PerceptionPanel, rviz_common::Panel)
