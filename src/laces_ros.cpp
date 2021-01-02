#include <laces/laces_ros.hpp>

#include <pluginlib/class_list_macros.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace laces {

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
    data_(init_data(_cm, _footprint)), cm_(&_cm) {}

std::pair<float, Eigen::Vector3d>
laces_ros::get_cost(const pose_msg& _msg) {
  if (!cm_)
    throw std::runtime_error("no costmap provided");

  // check if the robot-pose is planar - otherwise we cannot really deal with it
  if (std::abs(_msg.orientation.x) > 1e-6 ||
      std::abs(_msg.orientation.y) > 1e-6)
    throw std::runtime_error("3-dimensional poses are not supported");

  // note: all computations are done in the cell space.
  // indedices are map (m), baselink (b) and kernel (k).
  const auto res = cm_->getResolution();
  if (res <= 0)
    throw std::runtime_error("resolution must be positive");

  // convert meters to cell-space
  const eg::Vector2d b_origin(_msg.position.x, _msg.position.y);
  const eg::Vector2d m_origin =
      (b_origin - eg::Vector2d(cm_->getOriginX(), cm_->getOriginY())) *
      (1. / res);

  // get the transform from map (m) to kernel (k)
  const transform_type m_to_b =
      to_eigen(m_origin.x(), m_origin.y(), tf2::getYaw(_msg.orientation));
  // todo check if minus is correct
  const transform_type b_to_k = to_eigen(data_.d.center.x, data_.d.center.y, 0);
  const transform_type m_to_k = m_to_b * b_to_k;

  const box_type k_kernel_bb = to_box(data_.edt);
  const box_type m_kernel_bb = m_to_k * k_kernel_bb;

  // dirty rounding: we have to remove negative values, so we can cast to
  // unsigned int below
  box_type cell_kernel_bb = m_kernel_bb.array().round().matrix();
  const std::array<double, 2> cm_size{
      static_cast<double>(cm_->getSizeInCellsX()),
      static_cast<double>(cm_->getSizeInCellsY())};
  // iterate over the rows
  for (int rr = 0; rr != cell_kernel_bb.rows(); ++rr) {
    // clamp everything between zero and map size
    cell_kernel_bb.row(rr) =
        cell_kernel_bb.row(rr).array().max(0).min(cm_size.at(rr)).matrix();
  }

  // now we can convert the union_box to map coordinates
  cm_polygon sparse_outline, dense_outline;
  sparse_outline.reserve(cell_kernel_bb.cols());
  for (int cc = 0; cc != cell_kernel_bb.cols(); ++cc) {
    const auto col = cell_kernel_bb.col(cc).cast<unsigned int>();
    sparse_outline.emplace_back(cm::MapLocation{col(0), col(1)});
  }

  // get the cells of the dense outline
  cm_->polygonOutlineCells(sparse_outline, dense_outline);

  // convert the cells into line-intervals
  std::unordered_map<size_t, cell_interval> lines;

  for (const auto& cell : dense_outline)
    lines[cell.y].extend(cell.x);

  float sum = 0;
  Eigen::Vector3d derivative = Eigen::Vector3d::Zero();
  const transform_type k_to_m = m_to_k.inverse();
  // todo if the kernel and the map don't overlap, we will have an error
  // iterate over all lines
  const auto char_map = cm_->getCharMap();
  for (const auto& line : lines) {
    // the interval's max is inclusive, so we increment it
    const auto end = cm_->getIndex(line.second.max, line.first) + 1;
    auto index = cm_->getIndex(line.second.min, line.first);

    // iterate over all cells within one line-intervals
    for (auto x = line.second.min; index != end; ++index, ++x) {
      // we only want lethal cells
      if (char_map[index] != costmap_2d::LETHAL_OBSTACLE)
        continue;

      // convert to the kernel frame
      const eg::Vector2d m_cell(x, line.first);
      const eg::Vector2i k_cell =
          (k_to_m * m_cell).array().round().cast<int>().matrix();

      // check if k_cell is valid
      if ((k_cell.array() < 0).any() || k_cell(0) >= data_.edt.cols ||
          k_cell(1) >= data_.edt.rows)
        continue;

      sum += data_.edt.at<float>(k_cell(1), k_cell(0));
      derivative(0) += data_.d.dx.at<float>(k_cell(1), k_cell(0));
      derivative(1) += data_.d.dy.at<float>(k_cell(1), k_cell(0));
      derivative(2) += data_.d.dtheta.at<float>(k_cell(1), k_cell(0));
    }
  }

  return {sum, derivative};
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