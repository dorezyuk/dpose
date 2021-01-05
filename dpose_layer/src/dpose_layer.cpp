#include <dpose_layer/dpose_layer.hpp>

#include <geometry_msgs/PoseStamped.h>
#include <pluginlib/class_list_macros.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

namespace dpose_layer {

using namespace dpose_core;
using geometry_msgs::PoseStamped;

void
DposeLayer::updateBounds(double robot_x, double robot_y, double robot_yaw,
                         double*, double*, double*, double*) {
  // we just set the pose
  const auto origin_x = layered_costmap_->getCostmap()->getOriginX();
  const auto origin_y = layered_costmap_->getCostmap()->getOriginY();
  const auto res = layered_costmap_->getCostmap()->getResolution();
  robot_pose_.x() = (robot_x - origin_x) / res;
  robot_pose_.y() = (robot_y - origin_y) / res;
  robot_pose_.z() = robot_yaw;
}

PoseStamped
to_pose(const Eigen::Vector3d _pose, const std::string& _frame) {
  PoseStamped msg;
  msg.header.frame_id = _frame;
  msg.pose.position.x = _pose.x();
  msg.pose.position.y = _pose.y();
  tf2::Quaternion q;
  q.setRPY(0, 0, _pose.z());
  msg.pose.orientation = tf2::toMsg(q);
  return msg;
}

void
DposeLayer::updateCosts(costmap_2d::Costmap2D& _map, int, int, int, int) {
  const auto res = opt_.solve(impl_, robot_pose_);
  ROS_INFO_STREAM("current cost " << res.first);
  const auto origin_x = layered_costmap_->getCostmap()->getOriginX();
  const auto origin_y = layered_costmap_->getCostmap()->getOriginY();
  const auto resolution = layered_costmap_->getCostmap()->getResolution();

  auto msg = to_pose(res.second, "map");
  msg.pose.position.x = msg.pose.position.x * resolution + origin_x;
  msg.pose.position.y = msg.pose.position.y * resolution + origin_y;
  d_pub_.publish(msg);
}

void
DposeLayer::onFootprintChanged() {
  ROS_INFO("[dpose]: updating footprint");
  impl_ = pose_gradient(*layered_costmap_, {3});
}

void
DposeLayer::onInitialize() {
  gradient_decent::parameter param;
  param.epsilon = 0.5;
  param.iter = 20;
  param.step_t = 2;
  param.step_r = 0.1;

  opt_ = gradient_decent(std::move(param));
  ros::NodeHandle nh("~");
  d_pub_ = nh.advertise<PoseStamped>("derivative", 1);
}
}  // namespace dpose_layer

PLUGINLIB_EXPORT_CLASS(dpose_layer::DposeLayer, costmap_2d::Layer)
