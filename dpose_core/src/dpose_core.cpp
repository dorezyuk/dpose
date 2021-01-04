#include <dpose_core/dpose_core.hpp>

#include <pluginlib/class_list_macros.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace dpose_core {
namespace internal {

cv::Mat
draw_polygon(const cell_vector_type& _cells, cell_type& _shift,
             const cell_type& _padding) {
  // we need an area - so at least three points
  if (_cells.size() < 3)
    throw std::invalid_argument("cells must define a valid area");

  // negative padding might result in a bad image.
  if (_padding.x < 0 || _padding.y < 0)
    throw std::invalid_argument("padding cannot be negative");

  cv::Rect2d bb = cv::boundingRect(_cells);

  // we cannot do much if the input does not result in a valid bounding box
  if (bb.empty())
    throw std::invalid_argument("bb cannot be empty");

  // apply padding to the bounding box
  cv::Rect2i bbi(static_cast<cell_type>(bb.tl()) - _padding,
                 static_cast<cell_type>(bb.br()) + _padding);
  // setup the image
  cv::Mat image(bbi.size(), cv::DataType<uint8_t>::type, cv::Scalar(0));

  // it may be that the _cells contain negative numbers - we cannot draw them
  // so we have to shift everything by the lower left corner of the bounding box
  _shift = bbi.tl();
  auto shifted_cells = _cells;
  for (auto& cell : shifted_cells)
    cell -= _shift;

  // draw the polygon into the image
  cv::polylines(image, shifted_cells, true, 255);

  return image;
}

inline bool
is_valid(const cell_type& _cell, const cv::Mat& _image) noexcept {
  return 0 <= _cell.x && _cell.x < _image.cols && 0 <= _cell.y &&
         _cell.y < _image.rows;
}

/**
 * @brief Helper to shift all cells by _shift
 *
 * @param _cells input array of cells
 * @param _shift by how much to shift the _cells
 */
cell_vector_type
shift_cells(const cell_vector_type& _cells, const cell_type& _shift) noexcept {
  cell_vector_type shifted = _cells;
  for (auto& cell : shifted)
    cell += _shift;

  return shifted;
}

cv::Mat
smoothen_edges(cv::InputArray _edt, const cell_vector_type& _cells) {
  // we will perform some operations to post-process out image
  const auto edt = _edt.getMat();

  // get the mask of the polygon defined by cells
  cv::Mat mask(edt.size(), cv::DataType<uint8_t>::type, cv::Scalar(0));

  // sadly we have to copy _cells here
  std::vector<cell_vector_type> input({_cells});
  cv::fillPoly(mask, input, cv::Scalar(255));

  cv::Mat smoothen(edt.size(), cv::DataType<float>::type, cv::Scalar(0));
  // paint the polygon
  // todo fix these constants
  cv::polylines(smoothen, input, true, cv::Scalar(2));
  cv::GaussianBlur(smoothen, smoothen, cv::Size(5, 5), 0);

  // copy the input
  cv::copyTo(edt, smoothen, mask);

  return smoothen;
}

/**
 * @brief Retuns the maximum euclidean distance from the cell to the image
 * corners
 *
 * @param _image the image
 * @param _cell the cell
 * @return maximum distance from the cell to the image corners
 */
double
max_distance(cv::InputArray _image, const cell_type& _cell) {
  const auto m = _image.getMat();
  // get the corners
  const std::array<cell_type, 4> corners{cell_type{0, 0}, cell_type{0, m.rows},
                                         cell_type{m.cols, 0},
                                         cell_type{m.cols, m.rows}};

  // get the closest distance
  auto dist = 0.;
  for (const auto& corner : corners)
    dist = std::max(cv::norm(corner - _cell), dist);

  return dist;
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

cv::Mat
angular_derivative(cv::InputArray _image, const cell_type& _center) {
  // todo - this is not the end, but we are too lazy to deal with it now.
  // we would have to shift the image later...
  if (_center.x > 0 || _center.y > 0)
    throw std::invalid_argument("invalid center cell");

  const cell_type center = -_center;

  // get the distance
  // to be really safe against numeric issues we cast to int and not size_t
  const auto distance = static_cast<int>(max_distance(_image, center));

  // init the output image
  cv::Mat output(_image.size(), cv::DataType<float>::type, cv::Scalar(0));
  cv::Mat source = _image.getMat();

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

/**
 * @brief helper function to get the derivative from an image
 */
derivatives
init_derivatives(cv::InputArray _image, const cell_type& _center) {
  derivatives d;

  // x and y derivatives are really easy...
  cv::Sobel(_image, d.dx, cv::DataType<float>::type, 1, 0, 3);
  cv::Sobel(_image, d.dy, cv::DataType<float>::type, 0, 1, 3);
  d.center = _center;
  d.dtheta = angular_derivative(_image, _center);
  return d;
}

}  // namespace internal

data
init_data(const cell_vector_type& _cells) {
  using namespace internal;
  const cell_type padding(2, 2);
  cell_type center;

  cv::Mat im1 = draw_polygon(_cells, center, padding);
  // we need to inverse the im1: cv::distanceTransform calculates the distance
  // to zeros not to max.
  cv::Mat inv(im1.rows, im1.cols, im1.type());
  cv::bitwise_not(im1, inv);

  // get the euclidean distance transform
  cv::Mat edt(im1.rows, im1.cols, cv::DataType<float>::type);
  cv::distanceTransform(inv, edt, cv::DIST_L2, cv::DIST_MASK_PRECISE);

  data out;
  out.edt = smoothen_edges(edt, shift_cells(_cells, -center));
  out.d = init_derivatives(out.edt, center);
  return out;
}

/**
 * @brief todo document me
 */
inline cost_type
get_derivative(const derivatives& _data, const cell_type& _cell) {
  if (!internal::is_valid(_cell, _data.dx))
    return {0, 0, 0};

  return {_data.dx.at<float>(_cell), _data.dy.at<float>(_cell),
          _data.dtheta.at<float>(_cell)};
}

cost_type
get_derivative(const derivatives& _data, const cell_vector_type& _cells) {
  cost_type out(0, 0, 0);

  for (const auto& cell : _cells)
    out += get_derivative(_data, cell);

  // norm the output
  // todo check this
  if (!_cells.empty())
    out = out / cost_type(_cells.size(), _cells.size(), _cells.size());
  return out;
}

/**
 * @brief TODO document me
 */
inline float
get_cost(const cv::Mat& _data, const cell_type& _cell) {
  // invalid cells are ok for us (the image might be rotated...)
  if (!internal::is_valid(_cell, _data))
    return 0;
  return _data.at<float>(_cell);
}

float
get_cost(const data& _data, const cell_vector_type& _cells) {
  // accumulate the cost over the entire cell vector
  const auto edt = _data.edt;
  return std::accumulate(_cells.begin(), _cells.end(), 0.f,
                         [&](float _sum, const cell_type& _cell) {
                           return _sum + get_cost(edt, _cell);
                         });
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

  // flip the derivate back to the original frame.
  // note: we dont do this for the "theta"-part
  Eigen::Vector3d m_derivative;
  m_derivative.segment(0, 2) = m_to_k.rotation() * derivative.segment(0, 2);
  m_derivative(2) = derivative(2);
  return {sum, m_derivative};
}

std::pair<float, Eigen::Vector3d>
gradient_decent::solve(const pose_gradient& _laces, const Eigen::Vector3d& _start,
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

}  // namespace dpose
