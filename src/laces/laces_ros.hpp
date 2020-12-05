#pragma once

#include <laces/laces.hpp>

#include <costmap_2d/costmap_2d.h>
#include <costmap_2d/layered_costmap.h>
#include <geometry_msgs/Point.h>

#include <vector>

namespace laces {

/// @brief ros-polygon type
using polygon_msg = std::vector<geometry_msgs::Point>;

struct laces_ros {
  laces_ros() = default;
  laces_ros(const costmap_2d::Costmap2D& _cm, const polygon_msg& _footprint);
  explicit laces_ros(const costmap_2d::LayeredCostmap& _lcm);

private:
  derivatives derivatives_;
  const costmap_2d::Costmap2D* cm_ = nullptr;
};

}  // namespace laces