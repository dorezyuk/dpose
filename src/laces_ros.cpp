#include <laces/laces_ros.hpp>

namespace laces {

// helpers
using point_msg = geometry_msgs::Point;

inline point_type
get_point(const point_msg& _msg) noexcept {
  return {_msg.x, _msg.y};
}

polygon_type::ring_type
get_ring(const polygon_msg& _msg) noexcept {
  // create the ring
  polygon_type::ring_type ring;
  ring.resize(_msg.size());

  // transform the message
  std::transform(_msg.begin(), _msg.end(), ring.begin(), get_point);

  return ring;
}

polygon_type
get_polygon(const polygon_msg& _msg) noexcept {
  polygon_type p;
  p.outer() = get_ring(_msg);

  // correct the polygon
  bg::correct(p);

  return p;
}

}  // namespace laces