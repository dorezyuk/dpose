#pragma once

#include <dpose/dpose.hpp>

// #include <gpp_interface/pre_planning_interface.hpp>
#include <costmap_2d/costmap_2d.h>
#include <costmap_2d/layer.h>
#include <costmap_2d/layered_costmap.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <Eigen/Dense>

#include <vector>

namespace dpose {

namespace gm = geometry_msgs;
namespace cm = costmap_2d;
namespace eg = Eigen;

using point_msg = gm::Point;
using pose_msg = gm::Pose;
using polygon_msg = std::vector<gm::Point>;

using transform_type = Eigen::Isometry2d;
using box_type = Eigen::Matrix<double, 2, 5>;

inline transform_type
to_eigen(double _x, double _y, double _yaw) noexcept {
  return Eigen::Translation2d(_x, _y) * Eigen::Rotation2Dd(_yaw);
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

template <typename _T>
box_type
to_box(const _T& _x, const _T& _y) noexcept {
  box_type box;
  const auto x = static_cast<_T>(_x);
  const auto y = static_cast<_T>(_y);
  // order does not matter that much here
  // clang-format off
  box << 0, x, x, 0, 0,
         0, 0, y, y, 0;
  // clang-format on
  return box;
}

inline box_type
to_box(const costmap_2d::Costmap2D& _cm) noexcept {
  return to_box(_cm.getSizeInCellsX(), _cm.getSizeInCellsY());
}

inline box_type
to_box(const cv::Mat& _cm) noexcept {
  return to_box(_cm.cols, _cm.rows);
}

cell_vector_type
to_cells(const polygon_msg& _footprint, double _resolution);

struct laces_ros {
  laces_ros() = default;
  laces_ros(costmap_2d::Costmap2D& _cm, const polygon_msg& _footprint);
  explicit laces_ros(costmap_2d::LayeredCostmap& _lcm);

  std::pair<float, Eigen::Vector3d>
  get_cost(const Eigen::Vector3d& _se2) const;

private:
  data data_;
  // promise not to alter the costmap, but this class does not have a
  // const-correctness concept
  mutable costmap_2d::Costmap2D* cm_ = nullptr;
};

struct gradient_decent {
  struct parameter {
    size_t iter;
    double step_t;
    double step_r;
    double epsilon;
  };

  static std::pair<float, Eigen::Vector3d>
  solve(const laces_ros& _laces, const Eigen::Vector3d& _start,
        const parameter& _param);
};

struct LacesLayer : public costmap_2d::Layer {
  void
  updateBounds(double robot_x, double robot_y, double robot_yaw, double*,
               double*, double*, double*) override;

  void
  updateCosts(costmap_2d::Costmap2D& map, int, int, int, int) override;

  void
  onFootprintChanged() override;

protected:
  void
  onInitialize() override;

private:
  ros::Publisher d_pub_;
  Eigen::Vector3d robot_pose_;
  laces_ros impl_;
  gradient_decent::parameter param_;
};

}  // namespace dpose