#ifndef PERCEPTION_PERCEPT_GRASP_PANEL_H
#define PERCEPTION_PERCEPT_GRASP_PANEL_H

#include <ros/ros.h>
#include <rviz/panel.h>
#include <QWidget>
#include <QLineEdit>
#include <QCheckBox>
#include <QPushButton>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QFormLayout>
#include <QGridLayout>
#include <QTableWidget>
#include <QHeaderView>

#include <std_msgs/String.h>
#include <perception/GraspObjectArray.h>
#include <perception/GraspObject.h>

namespace perception
{

class PerceptGraspPanel : public rviz::Panel
{
  Q_OBJECT
public:
  PerceptGraspPanel(QWidget* parent = nullptr);
  virtual ~PerceptGraspPanel();

  virtual void load(const rviz::Config& config);
  virtual void save(rviz::Config config) const;

protected Q_SLOTS:
  void onApplyClicked();

protected:
  void graspObjectsCallback(const perception::GraspObjectArray::ConstPtr& msg);
  void updateChosenObjectDisplay(const perception::GraspObject& obj);
  void updateAllObjectsTable(const perception::GraspObjectArray::ConstPtr& msg);

  // ROS
  ros::NodeHandle nh_;
  ros::Publisher prompt_pub_;
  ros::Subscriber grasp_objects_sub_;

  // Config widgets
  QLineEdit* prompt_edit_;
  QCheckBox* enable_cdm_checkbox_;
  QPushButton* apply_btn_;

  // Status display widgets
  QLabel* current_prompt_label_;
  QLabel* object_count_label_;
  QLabel* detect_time_label_;

  // Chosen Object display widgets
  QGroupBox* chosen_object_group_;
  QLabel* chosen_id_label_;
  QLabel* chosen_class_label_;
  QLabel* chosen_bbox_label_;
  QLabel* chosen_3d_label_;
  QLabel* chosen_uv_label_;
  QLabel* chosen_depth_label_;
  QLabel* chosen_grasp_uv_label_;
  QLabel* chosen_width_label_;
  QLabel* chosen_angle_label_;
  QLabel* chosen_det_conf_label_;
  QLabel* chosen_grasp_aff_label_;

  // All Objects display widgets
  QGroupBox* all_objects_group_;
  QTableWidget* all_objects_table_;

  // Current data
  QString current_prompt_;
  int last_object_count_;
  double last_detect_time_;
};

}  // namespace perception

#endif  // PERCEPTION_PERCEPT_GRASP_PANEL_H
