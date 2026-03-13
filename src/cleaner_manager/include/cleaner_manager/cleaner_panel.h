/**
 * @file cleaner_panel.h
 * @brief RViz2 Panel for manual cleaner control.
 *
 * Displays: state, chassis status, arm status, selected target.
 * Actions: GO / CANCEL via /manual_control_node/command service.
 * No "Detect" button — object selection is handled by ObjectSelectorNode
 * (3D interactive markers) and PerceptionPanel (2D list).
 */

#ifndef CLEANER_MANAGER__CLEANER_PANEL_H
#define CLEANER_MANAGER__CLEANER_PANEL_H

#include <QWidget>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QProgressBar>
#include <QGroupBox>
#include <QListWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QTimer>
#include <QScrollArea>

#include <rclcpp/rclcpp.hpp>
#include <rviz_common/panel.hpp>
#include <rviz_common/display_context.hpp>

#include <cleaner_manager/msg/manual_status.hpp>
#include <cleaner_manager/msg/cleaner_manager_status.hpp>
#include <tracer_msgs/msg/tracer_status.hpp>
#include <piper_msgs/msg/piper_status.hpp>
#include <cleaner_manager/srv/manual_command.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <perception/msg/perception_config.hpp>

#include <atomic>
#include <mutex>
#include <memory>
#include <vector>
#include <string>

namespace cleaner_manager
{

struct TargetInfo {
    std::string category;
    double pos_x{0}, pos_y{0};
    double score{0};
    uint8_t status{0};   // 0=ACTIVE 1=PICKED 2=FAILED
    uint32_t observations{0};
    uint32_t attempts{0};
    float last_seen_age{0};
    bool viable{false};
};

class CleanerPanel : public rviz_common::Panel
{
    Q_OBJECT

public:
    explicit CleanerPanel(QWidget* parent = nullptr);
    virtual ~CleanerPanel();

    virtual void onInitialize() override;
    virtual void save(rviz_common::Config config) const override;
    virtual void load(const rviz_common::Config& config) override;

Q_SIGNALS:
    void statusUpdated(
        uint8_t state,
        const QString& state_name,
        const QString& message,
        const QString& target_category,
        const QString& target_object_id,
        double target_distance,
        double target_x, double target_y,
        double robot_x, double robot_y,
        const QString& grasp_step,
        double grasp_progress);

    void chassisUpdated(
        double linear_vel,
        double angular_vel,
        double battery_voltage,
        uint8_t control_mode,
        uint16_t error_code);

    void armUpdated(
        bool connected,
        bool enabled,
        const QString& error_state);

    void cleanerManagerStatusUpdated(
        uint8_t state,
        const QString& state_name,
        uint32_t targets_total,
        uint32_t targets_picked,
        uint32_t targets_failed,
        uint32_t targets_remaining,
        const QString& error_message,
        const QString& current_target_category);

protected Q_SLOTS:
    void handleStatusUpdated(
        uint8_t state,
        const QString& state_name,
        const QString& message,
        const QString& target_category,
        const QString& target_object_id,
        double target_distance,
        double target_x, double target_y,
        double robot_x, double robot_y,
        const QString& grasp_step,
        double grasp_progress);

    void handleChassisUpdated(
        double linear_vel,
        double angular_vel,
        double battery_voltage,
        uint8_t control_mode,
        uint16_t error_code);

    void handleArmUpdated(
        bool connected,
        bool enabled,
        const QString& error_state);

    void handleCleanerManagerStatusUpdated(
        uint8_t state,
        const QString& state_name,
        uint32_t targets_total,
        uint32_t targets_picked,
        uint32_t targets_failed,
        uint32_t targets_remaining,
        const QString& error_message,
        const QString& current_target_category);

    void onGoClicked();
    void onCancelClicked();
    void onStartAutoClicked();
    void onStopAutoClicked();
    void onResetPoolClicked();
    void onApplyGraspClicked();
    void spinOnce();

private:
    void setupUi();
    void setupRos();

    void statusCb(const cleaner_manager::msg::ManualStatus::SharedPtr msg);
    void chassisCb(const tracer_msgs::msg::TracerStatus::SharedPtr msg);
    void armCb(const piper_msgs::msg::PiperStatus::SharedPtr msg);
    void cleanerManagerStatusCb(const cleaner_manager::msg::CleanerManagerStatus::SharedPtr msg);

    void callCommand(uint8_t cmd);
    void updateTargetListUI();
    void publishGraspConfig();
    static QString stateColor(uint8_t state);
    static QString autoStateColor(uint8_t state);

    // === ROS ===
    rclcpp::Node::SharedPtr node_;
    rclcpp::executors::SingleThreadedExecutor::SharedPtr executor_;
    QTimer* spin_timer_{nullptr};

    rclcpp::Subscription<cleaner_manager::msg::ManualStatus>::SharedPtr sub_status_;
    rclcpp::Subscription<tracer_msgs::msg::TracerStatus>::SharedPtr sub_chassis_;
    rclcpp::Subscription<piper_msgs::msg::PiperStatus>::SharedPtr sub_arm_;
    rclcpp::Subscription<cleaner_manager::msg::CleanerManagerStatus>::SharedPtr sub_auto_status_;
    rclcpp::Client<cleaner_manager::srv::ManualCommand>::SharedPtr cmd_cli_;
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr start_auto_cli_;
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr stop_auto_cli_;
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr reset_pool_cli_;
    rclcpp::Publisher<perception::msg::PerceptionConfig>::SharedPtr pub_grasp_config_;

    // === UI ===
    QLabel* lbl_state_;
    QLabel* lbl_message_;

    // Chassis
    QLabel* lbl_chassis_status_;
    QLabel* lbl_chassis_vel_;
    QLabel* lbl_battery_;

    // Arm
    QLabel* lbl_arm_status_;

    // Target pool
    QLabel* lbl_current_target_;
    QListWidget* list_targets_;

    // Thread-safe target pool cache (dirty-flag pattern)
    std::mutex targets_mutex_;
    std::vector<TargetInfo> cached_targets_;
    std::atomic<bool> targets_dirty_{false};

    // Progress
    QGroupBox* grp_progress_;
    QProgressBar* bar_progress_;
    QLabel* lbl_grasp_step_;

    // Autonomous mode
    QGroupBox* grp_auto_;
    QLabel* lbl_auto_state_;
    QLabel* lbl_auto_stats_;
    QLabel* lbl_auto_error_;
    QPushButton* btn_start_auto_;
    QPushButton* btn_stop_auto_;
    QPushButton* btn_reset_pool_;

    // Grasp filter
    QLineEdit* edit_grasp_z_min_;
    QLineEdit* edit_grasp_z_max_;
    QLineEdit* edit_grasp_dist_max_;
    QLineEdit* edit_grasp_size_min_;
    QLineEdit* edit_grasp_size_max_;
    QPushButton* btn_apply_grasp_;

    // Buttons
    QPushButton* btn_go_;
    QPushButton* btn_cancel_;

    std::atomic<bool> panel_alive_{true};
    uint8_t current_state_{0};
    uint8_t auto_state_{0};
};

}  // namespace cleaner_manager

#endif  // CLEANER_MANAGER__CLEANER_PANEL_H
