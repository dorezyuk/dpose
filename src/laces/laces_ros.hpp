#pragma once

#include <laces/laces.hpp>

#include <costmap_2d/layered_costmap.h>
#include <geometry_msgs/Point.h>

#include <vector>

namespace laces {

// io for converting to and from ros

/// @brief ros-polygon type
using polygon_msg = std::vector<geometry_msgs::Point>;

/**
 * @brief Converts ros-polygons to boost::geometry::model::polygon
 *
 * @param _msg a ros-message
 * @return polygon_type our internal representation
 */
polygon_type
get_polygon(const polygon_msg& _msg) noexcept;

}  // namespace laces