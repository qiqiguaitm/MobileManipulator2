#include "perception/percept_3d_panel.h"
#include <pluginlib/class_list_macros.h>
#include <QTimer>
#include <cmath>

namespace perception
{

Percept3DPanel::Percept3DPanel(QWidget* parent)
  : rviz::Panel(parent)
  , status_received_(false)
{
  // Main layout
  QVBoxLayout* main_layout = new QVBoxLayout;

  // ========== Config Group ==========
  QGroupBox* config_group = new QGroupBox("Detection Config");
  QFormLayout* config_layout = new QFormLayout;

  // Prompt input
  prompt_edit_ = new QLineEdit;
  prompt_edit_->setPlaceholderText("e.g., box.bottle.cup");
  config_layout->addRow("Prompt:", prompt_edit_);

  // Min Score spinner
  min_score_spin_ = new QDoubleSpinBox;
  min_score_spin_->setRange(0.0, 1.0);
  min_score_spin_->setSingleStep(0.05);
  min_score_spin_->setValue(0.25);
  min_score_spin_->setDecimals(2);
  config_layout->addRow("Min Score:", min_score_spin_);

  // IoU Threshold spinner
  iou_threshold_spin_ = new QDoubleSpinBox;
  iou_threshold_spin_->setRange(0.0, 1.0);
  iou_threshold_spin_->setSingleStep(0.05);
  iou_threshold_spin_->setValue(0.5);
  iou_threshold_spin_->setDecimals(2);
  config_layout->addRow("IoU Threshold:", iou_threshold_spin_);

  // Apply button
  apply_btn_ = new QPushButton("Apply");
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

  current_min_score_label_ = new QLabel("-");
  status_layout->addRow("Min Score:", current_min_score_label_);

  current_iou_label_ = new QLabel("-");
  status_layout->addRow("IoU Threshold:", current_iou_label_);

  status_group->setLayout(status_layout);
  main_layout->addWidget(status_group);

  // ========== Results Group ==========
  QGroupBox* results_group = new QGroupBox("Detection Results");
  QFormLayout* results_layout = new QFormLayout;

  object_count_label_ = new QLabel("-");
  results_layout->addRow("Objects:", object_count_label_);

  categories_label_ = new QLabel("-");
  categories_label_->setWordWrap(true);
  results_layout->addRow("Categories:", categories_label_);

  detect_time_label_ = new QLabel("-");
  results_layout->addRow("Detect Time:", detect_time_label_);

  results_group->setLayout(results_layout);
  main_layout->addWidget(results_group);

  // Add stretch at bottom
  main_layout->addStretch();

  setLayout(main_layout);

  // Connect signals
  connect(apply_btn_, SIGNAL(clicked()), this, SLOT(onApplyClicked()));

  // ROS setup
  config_pub_ = nh_.advertise<perception::PerceptionConfig>("/perception/config", 1);
  status_sub_ = nh_.subscribe("/perception/status", 1, &Percept3DPanel::statusCallback, this);

  // Timer for updating display
  QTimer* timer = new QTimer(this);
  connect(timer, &QTimer::timeout, this, &Percept3DPanel::updateStatusDisplay);
  timer->start(500);  // Update every 500ms
}

Percept3DPanel::~Percept3DPanel()
{
}

void Percept3DPanel::onApplyClicked()
{
  perception::PerceptionConfig msg;
  msg.prompt = prompt_edit_->text().toStdString();
  msg.min_score = static_cast<float>(min_score_spin_->value());
  msg.iou_threshold = static_cast<float>(iou_threshold_spin_->value());

  config_pub_.publish(msg);
  ROS_INFO("PerceptPanel: Applied config - prompt='%s', min_score=%.2f, iou=%.2f",
           msg.prompt.c_str(), msg.min_score, msg.iou_threshold);
}

void Percept3DPanel::statusCallback(const perception::PerceptionStatus::ConstPtr& msg)
{
  last_status_ = *msg;
  status_received_ = true;
}

void Percept3DPanel::updateStatusDisplay()
{
  if (!status_received_) {
    return;
  }

  // Update current config display
  current_prompt_label_->setText(QString::fromStdString(last_status_.prompt));
  current_min_score_label_->setText(QString::number(last_status_.min_score, 'f', 2));
  current_iou_label_->setText(QString::number(last_status_.iou_threshold, 'f', 2));

  // Update detection results
  object_count_label_->setText(QString::number(last_status_.object_count));

  // Format categories
  QString categories;
  for (size_t i = 0; i < last_status_.categories.size(); ++i) {
    if (i > 0) categories += ", ";
    categories += QString::fromStdString(last_status_.categories[i]);
  }
  if (categories.isEmpty()) {
    categories = "-";
  }
  categories_label_->setText(categories);

  // Format detect time
  detect_time_label_->setText(QString::number(last_status_.last_detect_time * 1000, 'f', 1) + " ms");
}

void Percept3DPanel::load(const rviz::Config& config)
{
  rviz::Panel::load(config);

  QString prompt;
  if (config.mapGetString("prompt", &prompt)) {
    prompt_edit_->setText(prompt);
  }

  float min_score;
  if (config.mapGetFloat("min_score", &min_score)) {
    min_score_spin_->setValue(min_score);
  }

  float iou_threshold;
  if (config.mapGetFloat("iou_threshold", &iou_threshold)) {
    iou_threshold_spin_->setValue(iou_threshold);
  }
}

void Percept3DPanel::save(rviz::Config config) const
{
  rviz::Panel::save(config);
  config.mapSetValue("prompt", prompt_edit_->text());
  config.mapSetValue("min_score", min_score_spin_->value());
  config.mapSetValue("iou_threshold", iou_threshold_spin_->value());
}

}  // namespace perception

PLUGINLIB_EXPORT_CLASS(perception::Percept3DPanel, rviz::Panel)
