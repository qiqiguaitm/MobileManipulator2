/**
 * @file perception_panel.h
 * @brief RViz2 Panel for perception status and real-time object selection.
 *
 * Threading: ROS callbacks update cached_objects_ under mutex + set objects_dirty_.
 *            spinOnce() runs on Qt main thread (10Hz timer) and flushes to UI.
 *            No std::vector in Qt signals — avoids qRegisterMetaType issues.
 */

#ifndef CLEANER_MANAGER__PERCEPTION_PANEL_H
#define CLEANER_MANAGER__PERCEPTION_PANEL_H

#include <QWidget>
#include <QLabel>
#include <QPushButton>
#include <QLineEdit>
#include <QListWidget>
#include <QGroupBox>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QTimer>
#include <QScrollArea>

#include <rclcpp/rclcpp.hpp>
#include <rviz_common/panel.hpp>
#include <rviz_common/display_context.hpp>

#include <std_msgs/msg/string.hpp>
#include <perception/msg/perception_status.hpp>
#include <perception/msg/object3_d_array.hpp>
#include <perception/msg/perception_config.hpp>
#include <cleaner_manager/srv/manual_command.hpp>

#include <atomic>
#include <mutex>
#include <memory>
#include <vector>

namespace cleaner_manager
{

// POD cache for object data
struct ObjectInfo {
    std::string object_id;
    std::string category;
    double distance;
    double score;
    std::string source;
    double pos_x, pos_y, pos_z;  // in source frame (base_link)
    double depth;                 // camera depth (m)
    double bbox_w, bbox_h;        // bbox pixel size
    double phys_w, phys_h;        // estimated physical size (m)
};

class PerceptionPanel : public rviz_common::Panel
{
    Q_OBJECT

public:
    explicit PerceptionPanel(QWidget* parent = nullptr);
    virtual ~PerceptionPanel();

    virtual void onInitialize() override;
    virtual void save(rviz_common::Config config) const override;
    virtual void load(const rviz_common::Config& config) override;

Q_SIGNALS:
    // Only primitive/Qt types here — safe for QueuedConnection across threads
    void perceptionStatusUpdated(
        const QString& prompt,
        float min_score,
        uint32_t object_count,
        float last_detect_time);

protected Q_SLOTS:
    void handlePerceptionStatusUpdated(
        const QString& prompt,
        float min_score,
        uint32_t object_count,
        float last_detect_time);

    void onApplyConfigClicked();
    void onObjectItemClicked(QListWidgetItem* item);
    void spinOnce();

private:
    void setupUi();
    void setupRos();
    void publishConfig();       // publish current UI settings to perception node
    void updateObjectListUI();  // called from spinOnce() on Qt main thread

    void perceptionStatusCb(const perception::msg::PerceptionStatus::SharedPtr msg);
    void objectsCb(const perception::msg::Object3DArray::SharedPtr msg);

    // === ROS ===
    rclcpp::Node::SharedPtr node_;
    rclcpp::executors::SingleThreadedExecutor::SharedPtr executor_;
    QTimer* spin_timer_{nullptr};

    rclcpp::Subscription<perception::msg::PerceptionStatus>::SharedPtr sub_perc_status_;
    rclcpp::Subscription<perception::msg::Object3DArray>::SharedPtr sub_objects_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_default_prompt_;
    rclcpp::Publisher<perception::msg::PerceptionConfig>::SharedPtr pub_config_;
    rclcpp::Client<cleaner_manager::srv::ManualCommand>::SharedPtr cmd_cli_;

    // === UI ===
    QLabel* lbl_perc_status_;
    QLabel* lbl_detect_time_;
    QLabel* lbl_object_count_;

    QLineEdit* edit_prompt_;
    QLineEdit* edit_min_score_;
    QPushButton* btn_apply_config_;

    QListWidget* list_objects_;

    // Thread-safe object cache: written by ROS cb, read by Qt main thread
    std::mutex objects_mutex_;
    std::vector<ObjectInfo> cached_objects_;
    std::atomic<bool> objects_dirty_{false};

    std::atomic<bool> panel_alive_{true};

    // Detect timeout tracking (accessed only from spinOnce / ROS cb)
    rclcpp::Time last_objects_time_;
    std::atomic<bool> perc_online_{false};
};

}  // namespace cleaner_manager

#endif  // CLEANER_MANAGER__PERCEPTION_PANEL_H
