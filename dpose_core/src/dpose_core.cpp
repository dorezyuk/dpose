#include <dpose_core/dpose_core.hpp>

#include <pluginlib/class_list_macros.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dpose_core {

/// @brief a closed rectangle
template <typename _T>
using rectangle_type = Eigen::Matrix<_T, 2ul, 5ul>;

/// @brief constructs a rectangle with the given width and height
/// @param w width of the rectangle
/// @param h height of the rectangle
template <typename _T>
rectangle_type<_T>
to_rectangle(const _T& w, const _T& h) noexcept {
  rectangle_type<_T> box;
  // order does not matter that much here
  // clang-format off
  box << 0, w, w, 0, 0,
         0, 0, h, h, 0;
  // clang-format on
  return box;
}

// inline box_type
// to_box(const costmap_2d::Costmap2D& _cm) noexcept {
//   return to_box(_cm.getSizeInCellsX(), _cm.getSizeInCellsY());
// }

namespace internal {

/// @brief computes a cost-image based on the polygon
/// @param _cells corners of a sparse polygon
/// @param _param additional parameters
cv::Mat
_get_cost(const cell_vector_type& _cells, const parameter& _param) {
  // get the bounding box
  const cv::Rect bb = cv::boundingRect(_cells);

  // apply our padding
  cv::Size bb_size = bb.size();
  bb_size.width += _param.padding;
  bb_size.height += _param.padding;

  // setup the image and draw the cells
  cv::Mat image(bb_size, cv::DataType<uint8_t>::type, cv::Scalar(255));
  cv::polylines(image, _cells, true, 0);

  // image cannot be just "white"
  assert(cv::countNonZero(image) == image.cols * image.rows &&
         "no polygon drawn");

  // get the euclidean distance transform
  cv::Mat edt(bb_size, cv::DataType<float>::type);
  cv::distanceTransform(image, edt, cv::DIST_L2, cv::DIST_MASK_PRECISE);

  assert(cv::countNonZero(edt) > 0 && "distance transform failed");

  // we now apply "smoothing" on the edges of the polygon. this means, we add
  // some gaussian blur beyond the real polygon - this helps later on in the
  // optimization.

  // get the mask of the polygon defined by cells
  image.setTo(cv::Scalar(0));
  std::vector<cell_vector_type> cells({_cells});
  cv::fillPoly(image, cells, cv::Scalar(255));

  assert(cv::countNonZero(image) > 0 && "filling of the mask failed");

  constexpr float offset = 1;
  const auto kernel_size = _param.padding * 2 + 1;

  // paint a blurry polygon
  cv::Mat smoothen(bb_size, cv::DataType<float>::type, cv::Scalar(0));
  cv::polylines(smoothen, cells, true, cv::Scalar(offset));
  cv::GaussianBlur(smoothen, smoothen, cv::Size(kernel_size, kernel_size), 0);

  // since the border of the polygon (which has within the edt the value zero)
  // has now the value offset, we need to "lift" all other costs by the offset.
  edt += offset;

  // copy the distance transform within the mast into final image
  cv::copyTo(edt, smoothen, image);

  return smoothen;
}

/**
 * @brief helper to prune repetitions from get_circular_cells
 *
 * @param _cells the pre-output from get_circular_cells
 */
void
unique_cells(cell_vector_type& _cells) noexcept {
  if (_cells.empty())
    return;
  // use first unique to remove the overlaps
  auto last = std::unique(_cells.begin(), _cells.end());

  // now check if we loop - see if the first cells reappears
  last = std::find(std::next(_cells.begin()), last, _cells.front());

  // drop the redundant data
  _cells.erase(last, _cells.end());
}

cell_vector_type
get_circular_cells(const cell_type& _center, size_t _radius) noexcept {
  // adjusted from
  // https://github.com/opencv/opencv/blob/master/modules/imgproc/src/drawing.cpp
  int x = _radius, y = 0;
  int err = 0, plus = 1, minus = (_radius << 1) - 1;

  std::array<cell_vector_type, 8> octets;

  while (x >= y) {
    // insert the octets - for now without fancy looping
    // note: the order of these octets is very important
    auto octet = octets.begin();
    // clang-format off
    octet->emplace_back(_center.x + x, _center.y + y); ++octet;
    octet->emplace_back(_center.x + y, _center.y + x); ++octet;
    octet->emplace_back(_center.x - y, _center.y + x); ++octet;
    octet->emplace_back(_center.x - x, _center.y + y); ++octet;
    octet->emplace_back(_center.x - x, _center.y - y); ++octet;
    octet->emplace_back(_center.x - y, _center.y - x); ++octet;
    octet->emplace_back(_center.x + y, _center.y - x); ++octet;
    octet->emplace_back(_center.x + x, _center.y - y); ++octet;
    // clang-format on

    ++y;
    err += plus;
    plus += 2;

    int mask = (err <= 0) - 1;

    err -= minus & mask;
    x += mask;
    minus -= mask & 2;
  }

  // now flatten the octets
  cell_vector_type cells;
  cells.reserve(octets.begin()->size() * octets.size());
  // we have to reverse every second octet
  bool reverse = false;
  for (const auto& octet : octets) {
    if (reverse)
      cells.insert(cells.end(), octet.rbegin(), octet.rend());
    else
      cells.insert(cells.end(), octet.begin(), octet.end());
    reverse = !reverse;
  }

  unique_cells(cells);
  return cells;
}

inline bool
is_valid(const cell_type& _cell, const cv::Mat& _image) noexcept {
  return 0 <= _cell.x && _cell.x < _image.cols && 0 <= _cell.y &&
         _cell.y < _image.rows;
}

/**
 * @brief calculates the angular gradient given a _prev and _next cell.
 *
 * It is expected that _image and _source have the same size.
 * The method is a helper for angular_derivative.
 *
 * @param _prev previous cell on a circle
 * @param _curr the cell of interest of a circle
 * @param _next next cell on a circle
 * @param _image image where to store the gradient
 * @param _source image based on which we are computing the gradient
 */
void
mark_gradient(const cell_type& _prev, const cell_type& _curr,
              const cell_type& _next, cv::Mat& _image,
              const cv::Mat& _source) noexcept {
  // skip if not all are valid
  if (is_valid(_curr, _image) && is_valid(_prev, _source) &&
      is_valid(_next, _source)) {
    _image.at<float>(_curr) =
        _source.at<float>(_next) - _source.at<float>(_prev);
  }
}

inline rectangle_type<int>
_to_rectangle(const cv::Mat& _cm) noexcept {
  return to_rectangle(_cm.cols, _cm.rows);
}

inline int
_max_distance(const cv::Mat& _image, const Eigen::Vector2i _cell) noexcept {
  const rectangle_type<int> r = _to_rectangle(_image) - _cell;
  const Eigen::Matrix<double, 1, 5ul> d = r.cast<double>().colwise().norm();
  return static_cast<int>(d.maxCoeff());
}

cv::Mat
angular_derivative(cv::InputArray _image, const Eigen::Vector2i& _center) {
  // get the distance
  const auto distance = _max_distance(_image.getMat(), _center);

  assert(distance >= 0 && "failed to compute max-distance");

  // init the output image
  cv::Mat output(_image.size(), cv::DataType<float>::type, cv::Scalar(0));
  cv::Mat source = _image.getMat();
  const cv::Point2i center(_center.x(), _center.y());

  // now iterate over the all steps
  for (int ii = 0; ii <= distance; ++ii) {
    const auto cells = get_circular_cells(center, ii);

    // now we loop over the cells and get the gradient
    // we will need at least three points for this
    if (cells.size() < 3)
      continue;

    // beginning and end are special
    mark_gradient(*cells.rbegin(), *cells.begin(), *std::next(cells.begin()),
                  output, source);

    // iterate over all consecutive cells
    for (auto prev = cells.begin(), curr = std::next(prev),
              next = std::next(curr);
         next != cells.end(); ++prev, ++curr, ++next)
      mark_gradient(*prev, *curr, *next, output, source);

    // now do the end
    mark_gradient(*std::next(cells.rbegin()), *cells.rbegin(), *cells.begin(),
                  output, source);
  }

  return output;
}

cell_vector_type
_to_open_cv(const polygon& _footprint) {
  cell_vector_type cells;
  cells.reserve(_footprint.cols());
  for (int cc = 0; cc != _footprint.cols(); ++cc)
    cells.emplace_back(_footprint(0, cc), _footprint(1, cc));

  return cells;
}

data
init_data(const polygon& _footprint, const parameter& _param) {
  // we need an area
  if (_footprint.cols() < 3)
    throw std::runtime_error("footprint must contain at least three points");

  data out;

  // get first the center
  const Eigen::Vector2i padding(_param.padding, _param.padding);
  out.center = _footprint.rowwise().minCoeff() - padding;

  // now shift everything by the center, so we just have positive values in the
  // footprint
  polygon footprint = _footprint - out.center;

  // some checks for the debuggers
  assert(footprint.array().minCoeff() == _param.padding &&
         "footprint shifting failed");

  // now convert the eigen-polygon to opencv-cells
  const auto cells = _to_open_cv(footprint);

  // get the cost image
  out.cost = _get_cost(cells, _param);

  // finally our three derivatives
  cv::Sobel(out.cost, out.d_x, cv::DataType<float>::type, 1, 0, 3);
  cv::Sobel(out.cost, out.d_y, cv::DataType<float>::type, 0, 1, 3);
  out.d_theta = angular_derivative(out.cost, out.center);

  return out;
}

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

}  // namespace internal

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

inline internal::data
init_data(const costmap_2d::Costmap2D& _cm, const polygon_msg& _footprint) {
  // return init_data(to_cells(_footprint, _cm.getResolution()));
}

pose_gradient::pose_gradient(costmap_2d::LayeredCostmap& _lcm) :
    pose_gradient(*_lcm.getCostmap(), _lcm.getFootprint()) {}

pose_gradient::pose_gradient(costmap_2d::Costmap2D& _cm,
                             const polygon_msg& _footprint) :
    data_(init_data(_cm, _footprint)), cm_(&_cm) {}

std::pair<float, Eigen::Vector3d>
pose_gradient::get_cost(const Eigen::Vector3d& _se2) const {
  if (!cm_)
    throw std::runtime_error("no costmap provided");

  // note: all computations are done in the cell space.
  // indedices are map (m), baselink (b) and kernel (k).
  // get the transform from map (m) to kernel (k)
  const transform_type m_to_b = to_eigen(_se2.x(), _se2.y(), _se2.z());
  const transform_type b_to_k = to_eigen(data_.center.x(), data_.center.y(), 0);
  const transform_type m_to_k = m_to_b * b_to_k;

  const box_type k_kernel_bb = to_box(data_.cost);
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
      if ((k_cell.array() < 0).any() || k_cell(0) >= data_.cost.cols ||
          k_cell(1) >= data_.cost.rows)
        continue;

      sum += data_.cost.at<float>(k_cell(1), k_cell(0));
      derivative(0) += data_.d_x.at<float>(k_cell(1), k_cell(0));
      derivative(1) += data_.d_y.at<float>(k_cell(1), k_cell(0));
      derivative(2) += data_.d_theta.at<float>(k_cell(1), k_cell(0));
    }
  }

  // flip the derivate back to the original frame.
  // note: we dont do this for the "theta"-part
  Eigen::Vector3d m_derivative;
  m_derivative.segment(0, 2) = m_to_k.rotation() * derivative.segment(0, 2);
  m_derivative(2) = derivative(2);
  return {sum, m_derivative};
}

std::pair<float, Eigen::Vector3d>
gradient_decent::solve(const pose_gradient& _laces,
                       const Eigen::Vector3d& _start,
                       const gradient_decent::parameter& _param) {
  // super simple gradient decent algorithm with a limit on the max step
  // for now we set it to 1 cell size.
  std::pair<float, Eigen::Vector3d> res{0.f, _start};
  for (size_t ii = 0; ii != _param.iter; ++ii) {
    // get the derivative (d)
    auto d = _laces.get_cost(res.second);

    // scale the vector such that its norm is at most the _param.step
    // (the scaling is done seperately for translation (t) and rotation (r))
    const auto norm_t = std::max(d.second.segment(0, 2).norm(), _param.step_t);
    const auto norm_r = std::max(std::abs(d.second(2)), _param.step_r);
    d.second.segment(0, 2) *= (_param.step_t / norm_t);
    d.second(2) *= (_param.step_r / norm_r);

    // the "gradient decent"
    res.second += d.second;
    res.first = d.first;
    if (res.first <= _param.epsilon)
      break;
  }
  return res;
}

}  // namespace dpose_core
