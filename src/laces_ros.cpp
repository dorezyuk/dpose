#include <laces/laces_ros.hpp>

#include <tf2/utils.h>

#include <boost/geometry/strategies/transform.hpp>
#include <boost/geometry/strategies/transform/matrix_transformers.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace laces {

namespace internal {

namespace {

// only cpp-defined helpers

/// @brief Converts a ros-point to a boost::geometry point
inline bg_point_type
to_bg_point(const polygon_msg::value_type& _point) noexcept {
  return {_point.x, _point.y};
}

/// @brief helper to convert a ros-polygon to a boost::geometry polygon
bg_polygon_type::ring_type
to_bg_ring(const polygon_msg& _msg) noexcept {
  bg_polygon_type::ring_type ring;
  ring.resize(_msg.size());
  std::transform(_msg.begin(), _msg.end(), ring.begin(), to_bg_point);
  return ring;
}

/// @brief convers a ros-polygon to a boost::geometry polygon
bg_polygon_type
to_bg_polygon(const polygon_msg& _msg) noexcept {
  bg_polygon_type polygon;
  polygon.outer() = to_bg_ring(_msg);

  // try to fix the geometry
  if (!bg::is_valid(polygon))
    bg::correct(polygon);

  return polygon;
}

/// @brief generates a polygon from the corners of a bounding box
bg_polygon_type
to_bg_polygon(const bg_box_type& _box) noexcept {
  // assemble the polygon
  bg_polygon_type polygon;
  // create the missing points
  const bg_point_type ul{_box.min_corner().x(), _box.max_corner().y()};
  const bg_point_type lr{_box.max_corner().x(), _box.min_corner().y()};
  // create the outer ring in CCW
  const bg_polygon_type::ring_type ring = {
      _box.min_corner(), ul, _box.max_corner(), lr, _box.min_corner()};

  polygon.outer() = ring;
  return polygon;
}

/// @brief generates a bounding box from a boost::geometry polygon
/// @throw std::runtime_error if the input is not valid
bg_box_type
to_bg_box(const bg_polygon_type& _polygon) {
  // we cannot do much if the input is not a valid polygon
  if (!bg::is_valid(_polygon))
    throw std::runtime_error("invalid polygon");

  // get the bounding box
  bg_box_type bb;
  bg::envelope(_polygon, bb);
  return bb;
}

}  // namespace

bg_box_type
to_bg_box(const polygon_msg& _msg) {
  const auto polygon = to_bg_polygon(_msg);
  return to_bg_box(polygon);
}

namespace {

// only cpp-defined helpers

/// @brief generates the bounding box from the costmap's bounds
bg_box_type
to_bg_box(const costmap_2d::Costmap2D& _cm) noexcept {
  // retrieve the end-points of the map
  const bg_point_type p1(_cm.getOriginX(), _cm.getOriginY());
  const bg_point_type p2(_cm.getOriginX() + _cm.getSizeInMetersX(),
                         _cm.getOriginY() + _cm.getSizeInMetersY());
  // it goes min_corner, max_corner
  return {p1, p2};
}

namespace bgst = bg::strategy::transform;

/// @brief boost-specific transformations
using bg_matrix_type = bgst::matrix_transformer<double, 2, 2>;
using bg_rotate_type = bgst::rotate_transformer<bg::radian, double, 2, 2>;
using bg_translate_type = bgst::translate_transformer<double, 2, 2>;

/// @brief returns the matrix transform from _x, _y and _yaw
inline bg_matrix_type
to_bg_matrix(double _x, double _y, double _yaw) noexcept {
  return bg_matrix_type(bg_translate_type{_x, _y}.matrix() *
                        bg_rotate_type{_yaw}.matrix());
}

/// @brief returns the matrix transform from pose_msg
inline bg_matrix_type
to_bg_matrix(const pose_msg& _msg) noexcept {
  return to_bg_matrix(_msg.position.x, _msg.position.y,
                      tf2::getYaw(_msg.orientation));
}

using cv_box_type = cv::Rect2i;

cell_type
to_cv_cell(const bg_point_type& _point, const costmap_2d::Costmap2D& _cm) {
  unsigned int x, y;
  if (!_cm.worldToMap(_point.x(), _point.y(), x, y))
    throw std::runtime_error("box outside of the costmap");
  return {static_cast<int>(x), static_cast<int>(y)};
}

inline cv_box_type
to_cv_box(const bg_box_type& _bg, const costmap_2d::Costmap2D& _cm) {
  return {to_cv_cell(_bg.min_corner(), _cm), to_cv_cell(_bg.max_corner(), _cm)};
}

}  // namespace

}  // namespace internal

// its just internal for the users, not for us
using namespace internal;

cell_vector_type
to_cells(const polygon_msg& _footprint, double _resolution) {
  // we have standards...
  if (_resolution <= 0)
    throw std::runtime_error("resolution must be positive");

  cell_vector_type cells(_footprint.size());
  std::transform(_footprint.begin(), _footprint.end(), cells.begin(),
                 [&](const polygon_msg::value_type& __msg) {
                   // round to avoid unfair casting
                   return cell_type{
                       static_cast<int>(std::round(__msg.x / _resolution)),
                       static_cast<int>(std::round(__msg.y / _resolution))};
                 });
  return cells;
}

inline data
init_data(const costmap_2d::Costmap2D& _cm, const polygon_msg& _footprint) {
  return init_data(to_cells(_footprint, _cm.getResolution()));
}

laces_ros::laces_ros(costmap_2d::LayeredCostmap& _lcm) :
    laces_ros(*_lcm.getCostmap(), _lcm.getFootprint()) {}

laces_ros::laces_ros(const costmap_2d::Costmap2D& _cm,
                     const polygon_msg& _footprint) :
    data_(init_data(_cm, _footprint)), bb_(to_bg_box(_footprint)), cm_(&_cm) {}

float
laces_ros::get_cost(const pose_msg& _msg) {
  // todo maybe throw maybe warn...
  if (!cm_)
    return 0;

  // get the transform from the _msg
  const auto trans = to_bg_matrix(_msg);

  // convert the box to a polygon
  const auto original_polygon = to_bg_polygon(bb_);
  bg_polygon_type transformed_polygon;
  if (!bg::transform(original_polygon, transformed_polygon, trans))
    throw std::runtime_error("failed to transform the polygon");

  // now get the bounding box from the transformed polygon
  // this may throw a std::runtime_error
  const auto robot_bb = to_bg_box(transformed_polygon);

  // remove the part of the robot_bb outside of the map
  const auto map_bb = to_bg_box(*cm_);

  // the entire robot is outside of the map
  if (!bg::intersects(robot_bb, map_bb))
    return 0;
  bg_box_type union_bb;
  bg::intersection(robot_bb, map_bb, union_bb);

  // now we can convert the union_box to map coordinates
  return 0;
}

}  // namespace laces