#pragma once

#include <laces/laces.hpp>

// #include <gpp_interface/pre_planning_interface.hpp>
#include <costmap_2d/costmap_2d.h>
#include <costmap_2d/layer.h>
#include <costmap_2d/layered_costmap.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseStamped.h>

#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometry.hpp>

#include <vector>

namespace laces {

// below some io data types for ros-interface
// contrary to the conversion withing this lib, we will use the prefix _msg
// to indicate that this data-types are ros-msgs

namespace gm = geometry_msgs;
namespace cm = costmap_2d;

using pose_msg = geometry_msgs::Pose;
using point_msg = geometry_msgs::Point;
using polygon_msg = std::vector<point_msg>;

namespace internal {

// short-cut to the namespace
namespace bg = boost::geometry;

// short-cuts to the used typess
using bg_point_type = bg::model::d2::point_xy<double>;
using bg_polygon_type = bg::model::polygon<bg_point_type>;
using bg_box_type = bg::model::box<bg_point_type>;

/// @brief returns the bounding box of the _msg
/// @throw std::runtime_error if the generation of the bg_box_type fails
bg_box_type
to_bg_box(const polygon_msg& _msg);

/// @brief returns the bounding box from the data
/// @throw std::runtime_error if the generation of the bg_box_type fails
bg_box_type
to_bg_box(const data& _data, double _resolution);

}  // namespace internal

struct laces_ros {
  laces_ros() = default;
  laces_ros(costmap_2d::Costmap2D& _cm, const polygon_msg& _footprint);
  explicit laces_ros(costmap_2d::LayeredCostmap& _lcm);

  float
  get_cost(const pose_msg& _msg);

  cost_type
  get_derivative(const pose_msg& _msg);

private:
  data data_;
  internal::bg_polygon_type bb_;
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