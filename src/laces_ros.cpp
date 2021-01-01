#include <laces/laces_ros.hpp>

#include <pluginlib/class_list_macros.h>
#include <tf2/utils.h>

#include <boost/geometry/algorithms/convert.hpp>
#include <boost/geometry/strategies/transform.hpp>
#include <boost/geometry/strategies/transform/matrix_transformers.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace laces {

namespace internal {

namespace {

inline void
convert(const cv::Mat& _mat, bg_box_type& _box) {
  _box.min_corner() = bg_point_type{0, 0};
  _box.max_corner().x(_mat.rows);
  _box.max_corner().y(_mat.cols);
}

inline void
convert(const costmap_2d::Costmap2D& _map, bg_box_type& _box) {
  _box.min_corner() = bg_point_type{0, 0};
  _box.max_corner().x(_map.getSizeInCellsX());
  _box.max_corner().y(_map.getSizeInCellsY());
}

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

/// @brief convert a boost::geometry point to cell_type using the costmap
/// @throw std::runtime_error if the _point is outside of the _cm
cell_type
to_cv_cell(const bg_point_type& _point, const costmap_2d::Costmap2D& _cm) {
  unsigned int x, y;
  if (!_cm.worldToMap(_point.x(), _point.y(), x, y))
    throw std::runtime_error("box outside of the costmap");
  return {static_cast<int>(x), static_cast<int>(y)};
}

/// @brief convert a boost::geometry box to cv_box_type using the costmap
/// @throw std::runtime_error if the _bg is not fully inside the _cm
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

using cm_polygon = std::vector<costmap_2d::MapLocation>;
using cm_map = costmap_2d::Costmap2D;

/**
 * @brief interval defined by [min, max].
 *
 * Use interval::extend in order to add values.
 */
template <typename _T>
struct interval {
  interval() :
      min{std::numeric_limits<_T>::max()},
      max{std::numeric_limits<_T>::lowest()} {}

  inline void
  extend(const _T& _v) noexcept {
    min = std::min(min, _v);
    max = std::max(max, _v);
  }

  _T min, max;
};

using cell_interval = interval<size_t>;

inline data
init_data(const costmap_2d::Costmap2D& _cm, const polygon_msg& _footprint) {
  return init_data(to_cells(_footprint, _cm.getResolution()));
}

laces_ros::laces_ros(costmap_2d::LayeredCostmap& _lcm) :
    laces_ros(*_lcm.getCostmap(), _lcm.getFootprint()) {}

laces_ros::laces_ros(costmap_2d::Costmap2D& _cm,
                     const polygon_msg& _footprint) :
    data_(init_data(_cm, _footprint)),
    // bb_(to_bg_polygon(to_bg_box(_footprint))),
    cm_(&_cm) {}

float
laces_ros::get_cost(const pose_msg& _msg) {
  // todo maybe throw maybe warn...
  if (!cm_)
    return 0;

  // note: all computations are done in the cell space.
  // indedices are map (m), baselink (b) and kernel (k).

  const auto res = cm_->getResolution();
  if (res <= 0)
    throw std::runtime_error("resolution must be positive");

  // convert meters to cell-space
  auto msg = _msg;
  msg.position.x /= res;
  msg.position.y /= res;
  msg.position.z /= res;

  // get the transform from map (m) to kernel (k)
  const auto m_to_b = to_bg_matrix(msg);
  const auto b_to_k = to_bg_matrix(-data_.d.center.x, -data_.d.center.y, 0);
  const auto m_to_k = bg_matrix_type{m_to_b.matrix() * b_to_k.matrix()};

  // convert the kernel into the map frame
  bg_polygon_type k_kernel_bb, m_kernel_bb, m_map_bb;
  bg_box_type _kernel_bb, _map_bb;

  // convert first everything to boxes
  convert(*cm_, _map_bb);
  convert(data_.edt, _kernel_bb);

  // now convert the box types to polygons
  bg::convert(_map_bb, m_map_bb);
  bg::convert(_kernel_bb, k_kernel_bb);

  // transform k_kernel_bb to map frame
  const auto k_to_m = bg_matrix_type{boost::qvm::inverse(m_to_k.matrix())};
  if (!bg::transform(k_kernel_bb, m_kernel_bb, k_to_m))
    throw std::runtime_error("failed to transform the polygon");

  // the entire robot is outside of the map
  std::vector<bg_polygon_type> intersection_bb;
  bg::intersection(m_kernel_bb, m_map_bb, intersection_bb);

  if (intersection_bb.empty())
    return 0;

  // now we can convert the union_box to map coordinates
  const auto& ring = intersection_bb.front().outer();
  cm_polygon sparse_outline(ring.size());
  std::transform(ring.begin(), ring.end(), sparse_outline.begin(),
                 [](const bg_point_type& __p) {
                   return costmap_2d::MapLocation{std::round(__p.x()),
                                                  std::round(__p.y())};
                 });

  // get the cells of the dense outline
  cm_polygon dense_outline;
  cm_->polygonOutlineCells(sparse_outline, dense_outline);

  // convert the cells into line-intervals
  std::unordered_map<size_t, cell_interval> lines;

  for (const auto& cell : dense_outline)
    lines[cell.y].extend(cell.x);

  // iterate over all lines
  bg_point_type k_point;
  for (const auto& line : lines) {
    // the interval's max is inclusive, so we increment it
    const auto end = cm_->getIndex(line.first, line.second.max) + 1;
    auto start = cm_->getIndex(line.first, line.second.min);

    // iterate over all cells within one line-intervals
    for (auto x = line.second.min; start != end; ++start, ++x) {
      // we only want lethal cells
      if (cm_->getCharMap()[start] != costmap_2d::LETHAL_OBSTACLE)
        continue;

      // convert to the kernel frame
      if(!bg::transform(bg_point_type{x, line.first}, k_point, m_to_k))
        continue;
      
      // now get the cost from k_point
      k_point.x(std::round(k_point.x()));
      k_point.y(std::round(k_point.y()));


      // check if k_point is valid
      // get the cost
    }
  }

  return 0;
}

void
LacesLayer::updateCosts(costmap_2d::Costmap2D& _map, int, int, int, int) {
  ROS_INFO("[laces]: staring");
}

void
LacesLayer::onFootprintChanged() {
  ROS_INFO("[laces]: updating footprint");
  data_ = init_data(*layered_costmap_->getCostmap(),
                    layered_costmap_->getFootprint());
}

}  // namespace laces

PLUGINLIB_EXPORT_CLASS(laces::LacesLayer, costmap_2d::Layer)