#include "perception/percept_grasp_panel.h"
#include <pluginlib/class_list_macros.h>
#include <QTimer>
#include <cmath>

namespace perception
{

PerceptGraspPanel::PerceptGraspPanel(QWidget* parent)
  : rviz::Panel(parent)
  , current_prompt_("pen.box.phone.bottle.toy")
  , last_object_count_(0)
  , last_detect_time_(0.0)
{
  // Main layout
  QVBoxLayout* main_layout = new QVBoxLayout;

  // ========== Config Group ==========
  QGroupBox* config_group = new QGroupBox("Grasp Detection Config");
  QFormLayout* config_layout = new QFormLayout;

  // Prompt input
  prompt_edit_ = new QLineEdit;
  prompt_edit_->setPlaceholderText("e.g., pen.box.phone.bottle.toy");
  prompt_edit_->setText("pen.box.phone.bottle.toy");
  config_layout->addRow("Prompt:", prompt_edit_);

  // Enable CDM checkbox
  enable_cdm_checkbox_ = new QCheckBox;
  enable_cdm_checkbox_->setChecked(true);
  config_layout->addRow("Enable CDM:", enable_cdm_checkbox_);

  // Apply button
  apply_btn_ = new QPushButton("Apply Prompt");
  apply_btn_->setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 5px; }");
  config_layout->addRow("", apply_btn_);

  config_group->setLayout(config_layout);
  main_layout->addWidget(config_group);

  // ========== Status Group ==========
  QGroupBox* status_group = new QGroupBox("Current Status");
  QFormLayout* status_layout = new QFormLayout;

  current_prompt_label_ = new QLabel("-");
  current_prompt_label_->setWordWrap(true);
  status_layout->addRow("Prompt:", current_prompt_label_);

  object_count_label_ = new QLabel("-");
  status_layout->addRow("Objects:", object_count_label_);

  detect_time_label_ = new QLabel("-");
  status_layout->addRow("Detect Time:", detect_time_label_);

  status_group->setLayout(status_layout);
  main_layout->addWidget(status_group);

  // ========== Chosen Object Group ==========
  chosen_object_group_ = new QGroupBox("CHOSEN GRASP OBJECT");
  chosen_object_group_->setVisible(false);  // Initially hidden
  QGridLayout* chosen_layout = new QGridLayout;
  chosen_layout->setSpacing(5);
  chosen_layout->setContentsMargins(8, 8, 8, 8);

  auto addChosenRow = [&](int row, int col, const QString& key, QLabel*& valueLabel) {
    QLabel* keyLabel = new QLabel(key + ":");
    keyLabel->setStyleSheet("font-weight: bold; color: #555; font-size: 10px;");
    valueLabel = new QLabel("--");
    valueLabel->setStyleSheet("color: #00AA00; font-family: monospace; font-size: 10px;");
    chosen_layout->addWidget(keyLabel, row, col * 2);
    chosen_layout->addWidget(valueLabel, row, col * 2 + 1);
  };

  // Layout: 2 columns x 6 rows
  addChosenRow(0, 0, "ID", chosen_id_label_);
  addChosenRow(0, 1, "Class", chosen_class_label_);
  addChosenRow(1, 0, "BBox", chosen_bbox_label_);
  chosen_layout->addWidget(new QLabel(""), 1, 2);
  addChosenRow(2, 0, "3D(xyz)", chosen_3d_label_);
  addChosenRow(2, 1, "Depth", chosen_depth_label_);
  addChosenRow(3, 0, "UV", chosen_uv_label_);
  addChosenRow(3, 1, "Grasp UV", chosen_grasp_uv_label_);
  addChosenRow(4, 0, "Width", chosen_width_label_);
  addChosenRow(4, 1, "Angle", chosen_angle_label_);
  addChosenRow(5, 0, "Det Conf", chosen_det_conf_label_);
  addChosenRow(5, 1, "Grasp Aff", chosen_grasp_aff_label_);

  chosen_object_group_->setLayout(chosen_layout);
  main_layout->addWidget(chosen_object_group_);

  // ========== All Objects Group ==========
  all_objects_group_ = new QGroupBox("ALL GRASP OBJECTS");
  QVBoxLayout* all_objects_layout = new QVBoxLayout;

  all_objects_table_ = new QTableWidget(0, 11);
  all_objects_table_->setHorizontalHeaderLabels({
    "ID", "Class", "BBox", "3D(x,y,z)", "UV",
    "Depth", "Grasp UV", "W(mm)", "Angle", "Det", "Aff"
  });
  all_objects_table_->setEditTriggers(QAbstractItemView::NoEditTriggers);
  all_objects_table_->setSelectionBehavior(QAbstractItemView::SelectRows);
  all_objects_table_->horizontalHeader()->setStretchLastSection(true);
  all_objects_table_->setAlternatingRowColors(true);
  all_objects_table_->setMaximumHeight(200);
  all_objects_table_->setStyleSheet(
    "QTableWidget { font-size: 9px; }"
    "QHeaderView::section { background-color: #4CAF50; color: white; font-weight: bold; font-size: 9px; padding: 2px; }"
  );

  all_objects_layout->addWidget(all_objects_table_);
  all_objects_group_->setLayout(all_objects_layout);
  main_layout->addWidget(all_objects_group_);

  // Add stretch at bottom
  main_layout->addStretch();

  setLayout(main_layout);

  // Connect signals
  connect(apply_btn_, SIGNAL(clicked()), this, SLOT(onApplyClicked()));

  // ROS setup
  prompt_pub_ = nh_.advertise<std_msgs::String>("/perception_grasp_node/prompt", 1);
  grasp_objects_sub_ = nh_.subscribe("/perception_grasp_node/result", 1,
                                      &PerceptGraspPanel::graspObjectsCallback, this);
}

PerceptGraspPanel::~PerceptGraspPanel()
{
}

void PerceptGraspPanel::onApplyClicked()
{
  std_msgs::String msg;
  msg.data = prompt_edit_->text().toStdString();

  prompt_pub_.publish(msg);
  current_prompt_ = prompt_edit_->text();
  current_prompt_label_->setText(current_prompt_);

  ROS_INFO("PerceptGraspPanel: Applied prompt='%s'", msg.data.c_str());
}

void PerceptGraspPanel::load(const rviz::Config& config)
{
  rviz::Panel::load(config);

  QString prompt;
  if (config.mapGetString("prompt", &prompt)) {
    prompt_edit_->setText(prompt);
  }

  bool enable_cdm;
  if (config.mapGetBool("enable_cdm", &enable_cdm)) {
    enable_cdm_checkbox_->setChecked(enable_cdm);
  }
}

void PerceptGraspPanel::save(rviz::Config config) const
{
  rviz::Panel::save(config);
  config.mapSetValue("prompt", prompt_edit_->text());
  config.mapSetValue("enable_cdm", enable_cdm_checkbox_->isChecked());
}

void PerceptGraspPanel::graspObjectsCallback(const perception::GraspObjectArray::ConstPtr& msg)
{
  // Update status
  last_object_count_ = msg->objects.size();
  object_count_label_->setText(QString::number(last_object_count_));

  // Update detect time (from first object if available)
  if (!msg->objects.empty()) {
    // Assuming detection_time is stored somewhere in the message
    // For now, we'll skip this or you can add a field to GraspObjectArray
  }

  // Update chosen object display
  if (msg->chosen_index >= 0 && msg->chosen_index < static_cast<int>(msg->objects.size()))
  {
    const auto& chosen = msg->objects[msg->chosen_index];
    updateChosenObjectDisplay(chosen);
    chosen_object_group_->setVisible(true);
  }
  else
  {
    chosen_object_group_->setVisible(false);
  }

  // Update all objects table
  updateAllObjectsTable(msg);
}

void PerceptGraspPanel::updateChosenObjectDisplay(const perception::GraspObject& obj)
{
  // ID & Class
  chosen_id_label_->setText(QString::fromStdString(obj.object_id));
  chosen_class_label_->setText(QString::fromStdString(obj.category));

  // BBox
  chosen_bbox_label_->setText(QString("[%1,%2,%3,%4]")
    .arg((int)obj.bbox[0]).arg((int)obj.bbox[1])
    .arg((int)obj.bbox[2]).arg((int)obj.bbox[3]));

  // 3D Position (meters)
  chosen_3d_label_->setText(QString("[%1,%2,%3]")
    .arg(obj.position.x, 0, 'f', 3)
    .arg(obj.position.y, 0, 'f', 3)
    .arg(obj.position.z, 0, 'f', 3));

  // UV & Depth
  chosen_uv_label_->setText(QString("[%1,%2]")
    .arg((int)obj.grasp_center[0]).arg((int)obj.grasp_center[1]));
  chosen_depth_label_->setText(QString("%1m").arg(obj.depth, 0, 'f', 3));

  // Grasp UV
  chosen_grasp_uv_label_->setText(QString("[%1,%2]")
    .arg((int)obj.grasp_center[0]).arg((int)obj.grasp_center[1]));

  // Width (convert to mm)
  chosen_width_label_->setText(QString("%1mm").arg(obj.grasp_width3d * 1000, 0, 'f', 1));

  // Angle (convert to degrees)
  double angle_deg = obj.grasp_angle * 180.0 / M_PI;
  chosen_angle_label_->setText(QString("%1°").arg(angle_deg, 0, 'f', 1));

  // Confidence
  chosen_det_conf_label_->setText(QString::number(obj.detection_score, 'f', 3));
  chosen_grasp_aff_label_->setText(QString::number(obj.grasp_score, 'f', 3));
}

void PerceptGraspPanel::updateAllObjectsTable(const perception::GraspObjectArray::ConstPtr& msg)
{
  // Clear old data
  all_objects_table_->setRowCount(0);

  for (size_t i = 0; i < msg->objects.size(); ++i)
  {
    const auto& obj = msg->objects[i];
    int row = all_objects_table_->rowCount();
    all_objects_table_->insertRow(row);

    bool is_chosen = (msg->chosen_index == static_cast<int>(i));

    // Highlight chosen row
    QBrush bgBrush = is_chosen ? QBrush(QColor(0, 200, 0, 80)) : QBrush();

    auto setItem = [&](int col, const QString& text) {
      QTableWidgetItem* item = new QTableWidgetItem(text);
      if (is_chosen) {
        item->setBackground(bgBrush);
        item->setForeground(QBrush(QColor(0, 100, 0)));
        QFont font = item->font();
        font.setBold(true);
        item->setFont(font);
      }
      all_objects_table_->setItem(row, col, item);
    };

    // Column 0: ID
    setItem(0, QString::fromStdString(obj.object_id));

    // Column 1: Class
    setItem(1, QString::fromStdString(obj.category));

    // Column 2: BBox
    setItem(2, QString("[%1,%2,%3,%4]")
      .arg((int)obj.bbox[0]).arg((int)obj.bbox[1])
      .arg((int)obj.bbox[2]).arg((int)obj.bbox[3]));

    // Column 3: 3D
    setItem(3, QString("[%1,%2,%3]")
      .arg(obj.position.x, 0, 'f', 3)
      .arg(obj.position.y, 0, 'f', 3)
      .arg(obj.position.z, 0, 'f', 3));

    // Column 4: UV
    setItem(4, QString("[%1,%2]")
      .arg((int)obj.grasp_center[0]).arg((int)obj.grasp_center[1]));

    // Column 5: Depth
    setItem(5, QString::number(obj.depth, 'f', 3));

    // Column 6: Grasp UV
    setItem(6, QString("[%1,%2]")
      .arg((int)obj.grasp_center[0]).arg((int)obj.grasp_center[1]));

    // Column 7: Width (mm)
    setItem(7, QString::number(obj.grasp_width3d * 1000, 'f', 1));

    // Column 8: Angle (degrees)
    setItem(8, QString::number(obj.grasp_angle * 180.0 / M_PI, 'f', 1));

    // Column 9: Det Conf
    setItem(9, QString::number(obj.detection_score, 'f', 2));

    // Column 10: Grasp Aff
    setItem(10, QString::number(obj.grasp_score, 'f', 2));
  }

  // Auto-resize columns
  all_objects_table_->resizeColumnsToContents();
}

}  // namespace perception

PLUGINLIB_EXPORT_CLASS(perception::PerceptGraspPanel, rviz::Panel)
