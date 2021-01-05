#include <dpose_core/dpose_core.hpp>

#include <pluginlib/class_list_macros.h>

#include <opencv2/imgproc.hpp>

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

/// @brief a closed rectangle (hence 5 columns)
template <typename _T>
using rectangle_type = Eigen::Matrix<_T, 2UL, 5UL>;

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

/// @brief contains our open-cv related operations
namespace internal {

/// @brief open-cv specific data-types
using cell_type = cv::Point2i;
using cell_vector_type = std::vector<cell_type>;

/// @brief converts eigen to open-cv data
cell_vector_type
_to_open_cv(const polygon& _footprint) noexcept {
  cell_vector_type cells;
  cells.reserve(_footprint.cols());
  for (int cc = 0; cc != _footprint.cols(); ++cc)
    cells.emplace_back(_footprint(0, cc), _footprint(1, cc));

  return cells;
}

/// @brief computes a cost-image based on the polygon
/// @param _cells corners of a sparse polygon
/// @param _param additional parameters
cv::Mat
_get_cost(const polygon& _fp, const parameter& _param) {
  // convert the eigen-polygon to opencv-cells
  const auto _cells = _to_open_cv(_fp);

  // get the bounding box
  const cv::Rect bb = cv::boundingRect(_cells);

  // we need distinct points
  assert(!bb.empty() && "bounding box cannot be empty");

  // apply our padding
  cv::Size bb_size = bb.size();
  bb_size.width += (_param.padding * 2);
  bb_size.height += (_param.padding * 2);

  // setup the image and draw the cells
  cv::Mat image(bb_size, cv::DataType<uint8_t>::type, cv::Scalar(255));
  cv::polylines(image, _cells, true, cv::Scalar(0));

  // image cannot be just "white"
  assert(cv::imwrite("/tmp/poly.jpg", image));
  assert(cv::countNonZero(image) != image.cols * image.rows &&
         "no polygon drawn");

  // get the euclidean distance transform
  cv::Mat edt(bb_size, cv::DataType<float>::type);
  cv::distanceTransform(image, edt, cv::DIST_L2, cv::DIST_MASK_PRECISE);

  assert(cv::imwrite("/tmp/edt.jpg", edt));
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

  // copy the distance transform within the mask into final image
  cv::copyTo(edt, smoothen, image);

  return smoothen;
}

/// @brief helper to prune repetitions from _get_circular_cells
/// @param _cells the pre-output from _get_circular_cells
void
_unique_cells(cell_vector_type& _cells) noexcept {
  if (_cells.empty())
    return;
  // use first unique to remove the overlaps
  auto last = std::unique(_cells.begin(), _cells.end());

  // now check if we loop - see if the first cells reappears
  last = std::find(std::next(_cells.begin()), last, _cells.front());

  // drop the redundant data
  _cells.erase(last, _cells.end());
}

/// @brief returns the cells around _center at the given _radius
/// @param _center the center cell
/// @param _radius the radius (also in cells)
cell_vector_type
_get_circular_cells(const cell_type& _center, size_t _radius) noexcept {
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

  _unique_cells(cells);
  return cells;
}

/// @brief checks if the _cell is within the _image
inline bool
_is_valid(const cell_type& _cell, const cv::Mat& _image) noexcept {
  return 0 <= _cell.x && _cell.x < _image.cols && 0 <= _cell.y &&
         _cell.y < _image.rows;
}

/**
 * @brief calculates the angular gradient given a _prev and _next cell.
 *
 * It is expected that _image and _source have the same size.
 * The method is a helper for _angular_derivative.
 *
 * @param _prev previous cell on a circle
 * @param _curr the cell of interest of a circle
 * @param _next next cell on a circle
 * @param _image image where to store the gradient
 * @param _source image based on which we are computing the gradient
 */
void
_mark_gradient(const cell_type& _prev, const cell_type& _curr,
               const cell_type& _next, cv::Mat& _image,
               const cv::Mat& _source) noexcept {
  // skip if not all are valid
  if (_is_valid(_curr, _image) && _is_valid(_prev, _source) &&
      _is_valid(_next, _source)) {
    _image.at<float>(_curr) =
        _source.at<float>(_next) - _source.at<float>(_prev);
  }
}

/// @brief generates a rectangle with the size of _cm
inline rectangle_type<int>
_to_rectangle(const cv::Mat& _cm) noexcept {
  return to_rectangle<int>(_cm.cols, _cm.rows);
}

/// @brief returns the max distance from _image's corners to _cell
inline int
_max_distance(const cv::Mat& _image, const Eigen::Vector2i& _cell) noexcept {
  const rectangle_type<int> r = _to_rectangle(_image).colwise() - _cell;
  const Eigen::Matrix<double, 1UL, 5UL> d = r.cast<double>().colwise().norm();
  return static_cast<int>(d.maxCoeff());
}

/// @brief calculates the "circular" derivate of _image around the _center cell
/// @param _image image on which to perform the derivative
/// @param _center center of rotation.
cv::Mat
_angular_derivative(cv::InputArray _image,
                    const Eigen::Vector2i& _center) noexcept {
  // get the distance
  const auto distance = _max_distance(_image.getMat(), _center);

  assert(distance >= 0 && "failed to compute max-distance");

  // init the output image
  cv::Mat output(_image.size(), cv::DataType<float>::type, cv::Scalar(0));
  cv::Mat source = _image.getMat();
  const cell_type center(_center.x(), _center.y());

  // now iterate over the all steps
  // todo we can actually start from 1 and drop the check below
  for (int ii = 0; ii <= distance; ++ii) {
    const auto cells = _get_circular_cells(center, ii);

    // now we loop over the cells and get the gradient
    // we will need at least three points for this
    if (cells.size() < 3)
      continue;

    // beginning and end are special
    _mark_gradient(*cells.rbegin(), *cells.begin(), *std::next(cells.begin()),
                   output, source);

    // iterate over all consecutive cells
    for (auto prev = cells.begin(), curr = std::next(prev),
              next = std::next(curr);
         next != cells.end(); ++prev, ++curr, ++next)
      _mark_gradient(*prev, *curr, *next, output, source);

    // now do the end
    _mark_gradient(*std::next(cells.rbegin()), *cells.rbegin(), *cells.begin(),
                   output, source);
  }

  return output;
}

// function is exposed to the user
data
init_data(const polygon& _footprint, const parameter& _param) {
  // we need an area
  if (_footprint.cols() < 3)
    throw std::runtime_error("footprint must contain at least three points");

  data out;

  // get first the center
  const Eigen::Vector2i padding(_param.padding, _param.padding);
  out.center = padding - _footprint.rowwise().minCoeff();

  // now shift everything by the center, so we just have positive values
  polygon footprint = _footprint.colwise() + out.center;
  assert(footprint.array().minCoeff() == static_cast<int>(_param.padding) &&
         "footprint shifting failed");

  // get the cost image
  out.cost = _get_cost(footprint, _param);

  // finally our three derivatives
  cv::Sobel(out.cost, out.d_x, cv::DataType<float>::type, 1, 0, 3);
  cv::Sobel(out.cost, out.d_y, cv::DataType<float>::type, 0, 1, 3);
  out.d_theta = _angular_derivative(out.cost, out.center);

  // safe the image if we are running in debug mode (and scale the images)
  assert(cv::imwrite("/tmp/cost.jpg", out.cost * 10));
  assert(cv::imwrite("/tmp/d_x.jpg", out.d_x * 10 + 100));
  assert(cv::imwrite("/tmp/d_y.jpg", out.d_y * 10 + 100));
  assert(cv::imwrite("/tmp/d_theta.jpg", out.d_theta * 10 + 100));
  return out;
}

data
_init_data(const costmap_2d::Costmap2D& _cm, const polygon_msg& _footprint,
           const parameter& _param) {
  const auto res = _cm.getResolution();
  if (res <= 0)
    throw std::runtime_error("resolution must be positive");

  // convert the message to a eigen-polygon
  polygon cells(2, _footprint.size());
  for (int cc = 0; cc != cells.cols(); ++cc) {
    cells(0, cc) = std::round(_footprint.at(cc).x / res);
    cells(1, cc) = std::round(_footprint.at(cc).y / res);
  }

  // call the actual impl
  return init_data(cells, _param);
}

}  // namespace internal

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
using transform_type = Eigen::Isometry2d;

inline transform_type
to_eigen(double _x, double _y, double _yaw) noexcept {
  return Eigen::Translation2d(_x, _y) * Eigen::Rotation2Dd(_yaw);
}

pose_gradient::pose_gradient(costmap_2d::Costmap2D& _cm,
                             const polygon_msg& _footprint,
                             const parameter& _param) :
    data_(internal::_init_data(_cm, _footprint, _param)), cm_(&_cm) {}

pose_gradient::pose_gradient(costmap_2d::LayeredCostmap& _lcm,
                             const parameter& _param) :
    pose_gradient(*_lcm.getCostmap(), _lcm.getFootprint(), _param) {}

std::pair<float, Eigen::Vector3d>
pose_gradient::get_cost(const Eigen::Vector3d& _se2) const {
  using rectangle_d = rectangle_type<double>;
  using cm_polygon = std::vector<costmap_2d::MapLocation>;
  using namespace internal;

  if (!cm_)
    throw std::runtime_error("no costmap provided");

  // note: all computations are done in the cell space.
  // indedices are map (m), baselink (b) and kernel (k).
  // get the transform from map (m) to kernel (k)
  const transform_type m_to_b = to_eigen(_se2.x(), _se2.y(), _se2.z());
  const transform_type b_to_k =
      to_eigen(-data_.center.x(), -data_.center.y(), 0);
  const transform_type m_to_k = m_to_b * b_to_k;

  const rectangle_d k_kernel_bb = _to_rectangle(data_.cost).cast<double>();
  const rectangle_d m_kernel_bb = m_to_k * k_kernel_bb;

  // dirty rounding: we have to remove negative values, so we can cast to
  // unsigned int below
  rectangle_d cell_kernel_bb = m_kernel_bb.array().round().matrix();
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
  // todo check if i can replace this with something faster
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
      const Eigen::Vector2d m_cell(x, line.first);
      const Eigen::Vector2i k_cell =
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

tolerance::sphere_tolerance::sphere_tolerance(double _rad) :
    rad_(std::abs(_rad)) {}

tolerance::box_tolerance::box_tolerance(const pose& _box) :
    box_(_box.array().abs().matrix()) {}

tolerance::box_tolerance::box_tolerance(double _size) :
    box_tolerance(pose(_size, _size)) {}

tolerance::tolerance() : impl_(new none_tolerance()) {}

tolerance::tolerance(const mode& _m, const Eigen::Vector2d& _center) {
  switch (_m) {
    case mode::NONE: impl_.reset(new none_tolerance()); break;
    case mode::SPHERE: impl_.reset(new sphere_tolerance(_center.norm())); break;
    case mode::BOX: impl_.reset(new box_tolerance(_center)); break;
  }
}

gradient_decent::gradient_decent(parameter&& _param) noexcept :
    param_(std::move(_param)) {}

std::pair<float, Eigen::Vector3d>
gradient_decent::solve(const pose_gradient& _pg,
                       const Eigen::Vector3d& _start) const {
  // super simple gradient decent algorithm with a limit on the max step
  // for now we set it to 1 cell size.
  std::pair<float, Eigen::Vector3d> res{0.0F, _start};
  for (size_t ii = 0; ii != param_.iter; ++ii) {
    // get the derivative (d)
    auto d = _pg.get_cost(res.second);

    // scale the vector such that its norm is at most the _param.step
    // (the scaling is done seperately for translation (t) and rotation (r))
    const auto norm_t = std::max(d.second.segment(0, 2).norm(), param_.step_t);
    const auto norm_r = std::max(std::abs(d.second(2)), param_.step_r);
    d.second.segment(0, 2) *= (param_.step_t / norm_t);
    d.second(2) *= (param_.step_r / norm_r);

    // the "gradient decent"
    res.second += d.second;
    res.first = d.first;
    if (res.first <= param_.epsilon ||
        !param_.tol.within(_start.segment(0, 2), res.second.segment(0, 2)))
      break;
  }
  return res;
}

}  // namespace dpose_core
