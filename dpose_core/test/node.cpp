#include <dpose_core/dpose_core.hpp>
#include <costmap_2d/footprint.h>
#include <geometry_msgs/PoseStamped.h>
#include <map_msgs/OccupancyGridUpdate.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>

using namespace dpose;

constexpr char UNKNOWN = -1;
constexpr char LETHAL = 100;

unsigned char
convert(unsigned char value) {
  if (value == UNKNOWN)
    return cm::FREE_SPACE;
  else if (value >= LETHAL)
    return cm::LETHAL_OBSTACLE;
  else {
    double scale = (double)value / LETHAL;
    return scale * cm::LETHAL_OBSTACLE;
  }
}

static Eigen::Vector3d
to_se2_in_map(const pose_msg& _pose, const cm::Costmap2D& _map) {
  // check if the robot-pose is planar - otherwise we cannot really deal with it
  if (std::abs(_pose.orientation.x) > 1e-6 ||
      std::abs(_pose.orientation.y) > 1e-6)
    throw std::runtime_error("3-dimensional poses are not supported");

  const auto res = _map.getResolution();
  if (res <= 0)
    throw std::runtime_error("resolution must be positive");

  // convert meters to cell-space
  const eg::Vector2d pose(_pose.position.x, _pose.position.y);
  const eg::Vector2d map(_map.getOriginX(), _map.getOriginY());
  const eg::Vector2d origin = (pose - map) * (1. / res);

  const auto yaw = tf2::getYaw(_pose.orientation);
  return eg::Vector3d{origin.x(), origin.y(), yaw};
}

gm::PoseStamped
to_pose(const Eigen::Vector3d _pose, const std::string& _frame) {
  gm::PoseStamped msg;
  msg.header.frame_id = _frame;
  const auto yaw = std::atan2(_pose.y(), _pose.x());
  tf2::Quaternion q;
  q.setRPY(0, 0, yaw);
  msg.pose.orientation = tf2::toMsg(q);
  return msg;
}

struct map_sub {
  map_sub() {
    ros::NodeHandle nh("/navigation/move_base_flex");
    fp_ = cm::makeFootprintFromParams(nh);
    ROS_INFO_STREAM("Footprint loaded with the size  " << fp_.size());

    map_sub_ = nh.subscribe("/navigation/move_base_flex/global_costmap/costmap",
                            1, &map_sub::map_callback, this);
    map_update_sub_ = nh.subscribe(
        "/navigation/move_base_flex/global_costmap/costmap_updates", 1,
        &map_sub::map_update_callback, this);
    odom_sub_ = nh.subscribe("/base_pose_ground_truth", 1,
                             &map_sub::odom_callback, this);
    d_pub_ = nh.advertise<gm::PoseStamped>("derivative", 1);
  }

  void
  map_callback(const nav_msgs::OccupancyGrid& _msg) {
    const auto& info = _msg.info;

    ROS_INFO_STREAM("Received map [" << info.width << ", " << info.height
                                     << "]");

    map_.resizeMap(info.width, info.height, info.resolution,
                   info.origin.position.x, info.origin.position.y);

    unsigned int index = 0;
    const auto char_map = map_.getCharMap();

    for (unsigned int i = 0; i < info.height; ++i)
      for (unsigned int j = 0; j < info.width; ++j, ++index)
        char_map[index] = convert(_msg.data[index]);

    impl_ = laces_ros(map_, fp_);
    ROS_INFO_STREAM("Done");
  }

  void
  map_update_callback(const map_msgs::OccupancyGridUpdate& _msg) {
    ROS_INFO_STREAM("Received update [" << _msg.width << ", " << _msg.height
                                        << "]");
    const auto end_y = _msg.height + _msg.y;
    const auto end_x = _msg.width + _msg.x;
    if (end_x > map_.getSizeInCellsX() || end_y > map_.getSizeInCellsY()) {
      ROS_ERROR_STREAM("Update bigger than the actual map");
      return;
    }

    const auto char_map = map_.getCharMap();
    const auto size_x = map_.getSizeInCellsX();

    unsigned int di = 0;
    for (unsigned int y = _msg.y; y != end_y; ++y) {
      unsigned int index = y * size_x;
      for (unsigned int x = _msg.x; x != end_x; ++x, ++index) {
        char_map[index] = convert(_msg.data[di++]);
      }
    }

    ROS_INFO_STREAM("Done");
  }

  void
  odom_callback(const nav_msgs::Odometry& _msg) {
    ROS_INFO_STREAM("New odometry msg: " << _msg.pose.pose.position.x << ", "
                                         << _msg.pose.pose.position.x);
    if (_msg.header.frame_id != "map") {
      ROS_ERROR("Odometry must be in the \"map\" frame");
      return;
    }

    const auto res = impl_.get_cost(to_se2_in_map(_msg.pose.pose, map_));
    ROS_INFO_STREAM("Cost is " << res.first);
    if (res.first) {
      d_pub_.publish(to_pose(res.second, "base_link"));
    }
  }

private:
  ros::Subscriber map_sub_, map_update_sub_, odom_sub_;
  ros::Publisher d_pub_;
  costmap_2d::Costmap2D map_;
  std::vector<geometry_msgs::Point> fp_;

  laces_ros impl_;
};

int
main(int argc, char** argv) {
  ros::init(argc, argv, "laces_node");
  ros::NodeHandle nh;
  map_sub m;
  ros::spin();
}
