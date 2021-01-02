#pragma once

#include <laces/laces.hpp>

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

namespace laces {

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
  get_cost(const pose_msg& _msg);

private:
  data data_;
  costmap_2d::Costmap2D* cm_ = nullptr;
};

struct LacesLayer : public costmap_2d::Layer {
  void
  updateBounds(double robot_x, double robot_y, double robot_yaw, double*,
               double*, double*, double*) override {
    // we just set the pose
    robot_x_ = robot_x;
    robot_y_ = robot_y;
    robot_yaw_ = robot_yaw;
  }

  void
  updateCosts(costmap_2d::Costmap2D& map, int, int, int, int) override;

  void
  onFootprintChanged() override;

private:
  double robot_x_, robot_y_, robot_yaw_;
  data data_;
};

}  // namespace laces