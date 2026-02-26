/**
 * @file piper_grasp_panel.h
 * @brief RViz2 Panel for Piper Grasp Control (ROS2 Version)
 *
 * Features: Status monitoring, basic control, one-click grasp
 *
 * Units:
 *   - Position: mm (X, Y, Z)
 *   - Angles: degrees (Roll, Pitch, Yaw, Joint angles)
 *   - Gripper: mm (0 = closed, GRIPPER_MAX_MM = fully open)
 */

#ifndef PIPER_GRASP_PANEL_H
#define PIPER_GRASP_PANEL_H

#include <QWidget>
#include <QPushButton>
#include <QLabel>
#include <QLineEdit>
#include <QProgressBar>
#include <QTimer>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QScrollArea>
#include <QRegularExpression>
#include <QTime>
#include <QMap>

// ROS2 headers
#include <rclcpp/rclcpp.hpp>
#include <rviz_common/panel.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

// Standard messages
#include <std_srvs/srv/trigger.hpp>

// piper_msgs (all messages consolidated here in ROS2)
#include <piper_msgs/msg/piper_status_msg.hpp>
#include <piper_msgs/msg/piper_status.hpp>
#include <piper_msgs/srv/go_zero.hpp>
#include <piper_msgs/srv/enable_enhanced.hpp>
#include <piper_msgs/srv/set_gripper.hpp>
#include <piper_msgs/srv/go_ready.hpp>
#include <piper_msgs/srv/observe.hpp>
#include <piper_msgs/action/piper_pick.hpp>
#include <piper_msgs/action/piper_place.hpp>

#include <atomic>
#include <mutex>
#include <memory>

namespace piper_grasp
{

/**
 * @enum LetsGraspState
 * @brief Let's Grasp state machine
 */
enum class LetsGraspState {
    IDLE,
    OBSERVING,
    PICKING,
    PLACING,
    CANCELLING,
    COMPLETE,
    ERROR,
    // Observe Only mode (don't trigger Pick)
    OBSERVE_ONLY,
    // Auto Clear continuous mode states
    AUTO_OBSERVING,
    AUTO_PICKING,
    AUTO_PLACING,
    AUTO_WAITING,
    AUTO_FINISHING
};

/**
 * @class PiperGraspPanel
 * @brief RViz2 Panel for Piper control
 */
class PiperGraspPanel : public rviz_common::Panel
{
    Q_OBJECT

public:
    explicit PiperGraspPanel(QWidget* parent = nullptr);
    virtual ~PiperGraspPanel();

    virtual void save(rviz_common::Config config) const override;
    virtual void load(const rviz_common::Config& config) override;

    // Called after panel is added to RViz
    virtual void onInitialize() override;

protected Q_SLOTS:
    // Control buttons
    void onStopClicked();
    void onEnableClicked();
    void onDisableClicked();
    void onReconnectClicked();
    void onClearErrorClicked();
    void onGoReadyClicked();
    void onGoZeroClicked();
    void onOpenGripperClicked();
    void onObserveClicked();

    // Grasp control
    void onLetsGraspClicked();
    void onCancelClicked();
    void onAutoClearClicked();
    void onEndClearClicked();
    void onStatisticsPanelToggled();

    // Signal handlers (thread-safe UI updates)
    void handleStatusUpdated(bool connected, bool enabled,
                             const QString& error_state,
                             const QVector<double>& position,
                             const QVector<double>& joints,
                             double gripper_pos);
    void handleArmStatusUpdated(uint8_t ctrl_mode, uint8_t motion_status,
                                int64_t err_code,
                                const QVector<bool>& joint_limits,
                                const QVector<bool>& joint_comms);
    void handleServiceResult(const QString& name, bool success, const QString& msg);
    void handleObserveResult(bool success, const QString& category, double score,
                             const QString& error_msg);
    void handlePickResult(bool success, const QString& message);
    void handlePlaceResult(bool success, const QString& message);
    void handlePickProgress(double progress, const QString& step);
    void handlePlaceProgress(double progress, const QString& step);

Q_SIGNALS:
    void statusUpdated(bool connected, bool enabled,
                       const QString& error_state,
                       const QVector<double>& position,
                       const QVector<double>& joints,
                       double gripper_pos);
    void armStatusUpdated(uint8_t ctrl_mode, uint8_t motion_status,
                          int64_t err_code,
                          const QVector<bool>& joint_limits,
                          const QVector<bool>& joint_comms);
    void serviceResult(const QString& name, bool success, const QString& msg);
    void observeResult(bool success, const QString& category, double score,
                       const QString& error_msg);
    void pickResult(bool success, const QString& message);
    void placeResult(bool success, const QString& message);
    void pickProgress(double progress, const QString& step);
    void placeProgress(double progress, const QString& step);

private:
    // Setup
    void setupUi();
    void setupRos();

    // ROS2 callbacks
    void statusCallback(const piper_msgs::msg::PiperStatus::SharedPtr msg);
    void armStatusCallback(const piper_msgs::msg::PiperStatusMsg::SharedPtr msg);

    // Action goal response callbacks
    void pickGoalResponseCallback(
        const rclcpp_action::ClientGoalHandle<piper_msgs::action::PiperPick>::SharedPtr& goal_handle);
    void pickFeedbackCallback(
        rclcpp_action::ClientGoalHandle<piper_msgs::action::PiperPick>::SharedPtr goal_handle,
        const std::shared_ptr<const piper_msgs::action::PiperPick::Feedback> feedback);
    void pickResultCallback(
        const rclcpp_action::ClientGoalHandle<piper_msgs::action::PiperPick>::WrappedResult& result);

    void placeGoalResponseCallback(
        const rclcpp_action::ClientGoalHandle<piper_msgs::action::PiperPlace>::SharedPtr& goal_handle);
    void placeFeedbackCallback(
        rclcpp_action::ClientGoalHandle<piper_msgs::action::PiperPlace>::SharedPtr goal_handle,
        const std::shared_ptr<const piper_msgs::action::PiperPlace::Feedback> feedback);
    void placeResultCallback(
        const rclcpp_action::ClientGoalHandle<piper_msgs::action::PiperPlace>::WrappedResult& result);

    // Helpers
    void updateConnectionDisplay(bool connected, bool enabled);
    void updateLetsGraspUI();
    void setLastStatus(const QString& msg, bool success);
    QString getModeText(uint8_t mode);
    QString getMotionText(uint8_t status);
    void showErrorMessage(const QString& errorInfo);

    // Auto Clear helpers
    void startNextAutoCycle();
    void updateAutoStatistics();
    void resetAutoStatistics();
    void checkAutoTerminationConditions();

    // Spin timer for ROS2
    void spinOnce();

    // === ROS2 ===
    rclcpp::Node::SharedPtr node_;
    rclcpp::executors::SingleThreadedExecutor::SharedPtr executor_;
    QTimer* spin_timer_;

    // Subscriptions
    rclcpp::Subscription<piper_msgs::msg::PiperStatus>::SharedPtr sub_status_;
    rclcpp::Subscription<piper_msgs::msg::PiperStatusMsg>::SharedPtr sub_arm_status_;

    // Service clients
    rclcpp::Client<piper_msgs::srv::EnableEnhanced>::SharedPtr srv_enable_;
    rclcpp::Client<piper_msgs::srv::GoReady>::SharedPtr srv_go_ready_;
    rclcpp::Client<piper_msgs::srv::GoZero>::SharedPtr srv_go_zero_;
    rclcpp::Client<piper_msgs::srv::SetGripper>::SharedPtr srv_set_gripper_;
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr srv_stop_;
    rclcpp::Client<piper_msgs::srv::Observe>::SharedPtr srv_observe_;

    // Action clients
    using PickAction = piper_msgs::action::PiperPick;
    using PlaceAction = piper_msgs::action::PiperPlace;
    using PickGoalHandle = rclcpp_action::ClientGoalHandle<PickAction>;
    using PlaceGoalHandle = rclcpp_action::ClientGoalHandle<PlaceAction>;

    rclcpp_action::Client<PickAction>::SharedPtr pick_client_;
    rclcpp_action::Client<PlaceAction>::SharedPtr place_client_;
    PickGoalHandle::SharedPtr current_pick_goal_;
    PlaceGoalHandle::SharedPtr current_place_goal_;

    // === UI Components ===

    // Top bar
    QLabel* label_connection_;
    QLabel* label_error_;
    QPushButton* btn_stop_;

    // Control buttons
    QPushButton* btn_enable_;
    QPushButton* btn_disable_;
    QPushButton* btn_reconnect_;
    QPushButton* btn_clear_error_;
    QPushButton* btn_go_ready_;
    QPushButton* btn_go_zero_;
    QPushButton* btn_open_gripper_;
    QPushButton* btn_observe_;

    // Grasp
    QLineEdit* edit_prompt_;
    QPushButton* btn_lets_grasp_;
    QPushButton* btn_cancel_;
    QPushButton* btn_auto_clear_;
    QPushButton* btn_end_clear_;
    QProgressBar* bar_progress_;
    QLabel* label_grasp_status_;
    QLabel* label_last_status_;
    QLabel* label_stage_info_;
    QLabel* label_error_display_;

    // Auto Clear statistics panel
    QGroupBox* group_statistics_;
    QPushButton* btn_stats_toggle_;
    QLabel* label_cycle_count_;
    QLabel* label_success_count_;
    QLabel* label_fail_count_;
    QLabel* label_success_rate_;
    QLabel* label_object_types_;
    QLabel* label_duration_;
    QLabel* label_avg_time_;

    // ARM DETAILS
    QGroupBox* group_details_;
    QLabel* label_mode_;
    QLabel* label_motion_;
    QLabel* label_err_code_;
    QLabel* label_joint_limits_[6];
    QLabel* label_joint_comms_[6];
    QLabel* label_position_;
    QLabel* label_joints_;
    QProgressBar* bar_gripper_;
    QLabel* label_gripper_value_;

    // === State ===
    QString current_prompt_;
    std::atomic<int> connection_state_{0};
    std::atomic<bool> observe_valid_{false};
    std::atomic<LetsGraspState> lets_grasp_state_{LetsGraspState::IDLE};
    QString lets_grasp_error_msg_;
    std::atomic<bool> panel_alive_{true};

    // Stage information storage
    QVector<double> current_position_;
    QString observe_category_;
    double observe_score_;
    QVector<double> target_grasp_pose_;
    QVector<double> current_target_grasp_;
    QVector<double> place_position_;

    // Error auto-clear timer
    QTimer* error_clear_timer_;

    // === Auto Clear state ===
    std::atomic<bool> auto_clear_mode_{false};
    std::atomic<bool> end_clear_requested_{false};
    QTimer* auto_cycle_timer_;

    // Auto Clear statistics
    int auto_cycle_count_ = 0;
    int auto_success_count_ = 0;
    int auto_fail_count_ = 0;
    int auto_consecutive_failures_ = 0;
    int auto_consecutive_empty_observes_ = 0;
    QMap<QString, int> auto_object_counts_;
    QTime auto_start_time_;
    QTime auto_cycle_start_time_;
    bool statistics_panel_expanded_ = false;

    // === Constants ===
    static constexpr double GRIPPER_MAX_MM = 70.0;
    static constexpr int ERROR_DISPLAY_DURATION_MS = 15000;
    static constexpr int CAMERA_SETTLE_TIME_MS = 300;
    static constexpr int AUTO_CYCLE_DELAY_SUCCESS_MS = 0;
    static constexpr int AUTO_CYCLE_DELAY_FAILURE_MS = 3000;
    static constexpr int AUTO_MAX_CONSECUTIVE_FAILURES = 3;
    static constexpr int AUTO_MAX_CONSECUTIVE_EMPTY = 3;
};

}  // namespace piper_grasp

#endif  // PIPER_GRASP_PANEL_H
