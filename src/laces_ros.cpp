#include <laces/laces_ros.hpp>

#include <algorithm>
#include <cmath>

namespace laces {

cell_vector_type
to_cells(const polygon_msg& _footprint, double _resolution) {
  // we have standards...
  if(_resolution <= 0)
    throw std::runtime_error("resolution must be positive");

  cell_vector_type cells(_footprint.size());
  std::transform(_footprint.begin(), _footprint.end(), cells.begin(),
                 [&](const polygon_msg::value_type& __msg) {
                   // round to avoid unfair casting
                   return cell_type{std::round(__msg.x / _resolution),
                                    std::round(__msg.y / _resolution)};
                 });
  return cells;
}


laces_ros::laces_ros(const costmap_2d::LayeredCostmap& _lcm) {}

laces_ros::laces_ros(const costmap_2d::Costmap2D& _cm,
                     const polygon_msg& _footprint) {}

}  // namespace laces