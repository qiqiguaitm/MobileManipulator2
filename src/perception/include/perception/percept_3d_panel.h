#ifndef PERCEPTION_PERCEPT_3D_PANEL_H
#define PERCEPTION_PERCEPT_3D_PANEL_H

#include <ros/ros.h>
#include <rviz/panel.h>
#include <QWidget>
#include <QLineEdit>
#include <QDoubleSpinBox>
#include <QPushButton>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QFormLayout>
#include <QGridLayout>
#include <QTableWidget>
#include <QHeaderView>

#include <perception/PerceptionConfig.h>
#include <perception/PerceptionStatus.h>

namespace perception
{

class Percept3DPanel : public rviz::Panel
{
  Q_OBJECT
public:
  Percept3DPanel(QWidget* parent = nullptr);
  virtual ~Percept3DPanel();

  virtual void load(const rviz::Config& config);
  virtual void save(rviz::Config config) const;

protected Q_SLOTS:
  void onApplyClicked();

protected:
  void statusCallback(const perception::PerceptionStatus::ConstPtr& msg);
  void updateStatusDisplay();

  // ROS
  ros::NodeHandle nh_;
  ros::Publisher config_pub_;
  ros::Subscriber status_sub_;

  // Config widgets
  QLineEdit* prompt_edit_;
  QDoubleSpinBox* min_score_spin_;
  QDoubleSpinBox* iou_threshold_spin_;
  QPushButton* apply_btn_;

  // Status display widgets
  QLabel* current_prompt_label_;
  QLabel* current_min_score_label_;
  QLabel* current_iou_label_;
  QLabel* object_count_label_;
  QLabel* categories_label_;
  QLabel* detect_time_label_;

  // Current status data
  perception::PerceptionStatus last_status_;
  bool status_received_;
};

}  // namespace perception

#endif  // PERCEPTION_PERCEPT_3D_PANEL_H
