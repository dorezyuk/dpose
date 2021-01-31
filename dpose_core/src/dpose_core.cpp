/*
 * MIT License
 *
 * Copyright (c) 2021 Dima Dorezyuk
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include <dpose_core/dpose_core.hpp>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <iterator>
#include <stdexcept>
#include <vector>

namespace dpose_core {

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
inline rectangle<int>
_to_rectangle(const cv::Mat& _cm) noexcept {
  return to_rectangle<int>(_cm.cols, _cm.rows);
}

/// @brief returns the max distance from _image's corners to _cell
inline int
_max_distance(const cv::Mat& _image, const Eigen::Vector2i& _cell) noexcept {
  const rectangle<int> r = _to_rectangle(_image).colwise() - _cell;
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
  for (int ii = 1; ii <= distance; ++ii) {
    const auto cells = _get_circular_cells(center, ii);

    // now we loop over the cells and get the gradient
    // we will need at least three points for this
    assert(cells.size() > 2 && "invalid circular cells");

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

/// @brief will generate the jacobian data based on the cost_data
jacobian_data
_init_jacobian(const cost_data& _data) {
  jacobian_data J;
  // get our three derivatives
  cv::Sobel(_data.cost, J.d_x, cv::DataType<float>::type, 1, 0, 5, 1. / 64.);
  cv::Sobel(_data.cost, J.d_y, cv::DataType<float>::type, 0, 1, 5, 1. / 64.);
  J.d_z = _angular_derivative(_data.cost, _data.center);

  // safe the jacobians if compiled in debug mode
  assert(cv::imwrite("/tmp/d_x.jpg", J.d_x * 10 + 100));
  assert(cv::imwrite("/tmp/d_y.jpg", J.d_y * 10 + 100));
  assert(cv::imwrite("/tmp/d_theta.jpg", J.d_z * 10 + 100));

  return J;
}

/// @brief will generate the hessian data based on the jacobian data
hessian_data
_init_hessian(const cost_data& _data, const jacobian_data& _J,
              const Eigen::Vector2i& _center) {
  hessian_data H;
  // todo this is crap
  // second derivative to x
  cv::Sobel(_J.d_x, H.d_x_x, cv::DataType<float>::type, 1, 0, 5, 1. / 4096.);
  cv::Sobel(_J.d_y, H.d_y_x, cv::DataType<float>::type, 1, 0, 5, 1. / 4096.);
  cv::Sobel(_J.d_z, H.d_z_x, cv::DataType<float>::type, 1, 0, 5, 1. / 4096.);

  // safe the hessians if compiled in debug mode
  assert(cv::imwrite("/tmp/d_x_x.jpg", H.d_x_x * 10 + 100));
  assert(cv::imwrite("/tmp/d_y_x.jpg", H.d_y_x * 10 + 100));
  assert(cv::imwrite("/tmp/d_theta_x.jpg", H.d_z_x * 10 + 100));

  // second derivative to y
  cv::Sobel(_J.d_x, H.d_x_y, cv::DataType<float>::type, 0, 1, 5, 1. / 4096.);
  cv::Sobel(_J.d_y, H.d_y_y, cv::DataType<float>::type, 0, 1, 5, 1. / 4096.);
  cv::Sobel(_J.d_z, H.d_z_y, cv::DataType<float>::type, 0, 1, 5, 1. / 4096.);

  // safe the hessians if compiled in debug mode
  assert(cv::imwrite("/tmp/d_x_y.jpg", H.d_x_y * 10 + 100));
  assert(cv::imwrite("/tmp/d_y_y.jpg", H.d_y_y * 10 + 100));
  assert(cv::imwrite("/tmp/d_theta_y.jpg", H.d_z_y * 10 + 100));

  // second derivative to theta
  H.d_x_z = _angular_derivative(_J.d_x, _center) * 1. / 64.;
  H.d_y_z = _angular_derivative(_J.d_y, _center) * 1. / 64.;
  H.d_z_z = _angular_derivative(_J.d_z, _center) * 1. / 64.;

  // safe the hessians if compiled in debug mode
  assert(cv::imwrite("/tmp/d_x_theta.jpg", H.d_x_z * 10 + 100));
  assert(cv::imwrite("/tmp/d_y_theta.jpg", H.d_y_z * 10 + 100));
  assert(cv::imwrite("/tmp/d_theta_theta.jpg", H.d_z_z * 10 + 100));

  return H;
}

// function is exposed to the user
data
init_data(const polygon& _footprint, const parameter& _param) {
  // we need an area
  if (_footprint.cols() < 3)
    throw std::runtime_error("footprint must contain at least three points");

  data out;

  // get the center
  const Eigen::Vector2i padding(_param.padding, _param.padding);
  out.core.center = padding - _footprint.rowwise().minCoeff();

  // shift everything by the center, so we just have positive values
  polygon footprint = _footprint.colwise() + out.core.center;
  assert(footprint.array().minCoeff() == static_cast<int>(_param.padding) &&
         "footprint shifting failed");

  // get the cost image
  out.core.cost = _get_cost(footprint, _param);
  // safe the image if we are running in debug mode (and scale the images)
  assert(cv::imwrite("/tmp/cost.jpg", out.core.cost * 10));

  // get the derivatives
  out.J = _init_jacobian(out.core);

  // the hessian data might be optional
  if (_param.generate_hessian)
    out.H = _init_hessian(out.core, out.J, out.core.center);

  return out;
}

}  // namespace internal

using transform_type = Eigen::Isometry2d;

inline transform_type
to_eigen(double _x, double _y, double _yaw) noexcept {
  return Eigen::Translation2d(_x, _y) * Eigen::Rotation2Dd(_yaw);
}

/// @brief checks if the _box is inside a rectangle starting at (0, 0) and
/// ending at _max
inline bool
is_inside(const Eigen::Vector2d& _max, const rectangle<double>& _box) noexcept {
  return (_box.array() >= 0).all() && (_box.row(0).array() < _max(0)).all() &&
         (_box.row(1).array() < _max(1)).all();
}

pose_gradient::pose_gradient(const polygon& _footprint,
                             const parameter& _param) :
    data_(internal::init_data(_footprint, _param)) {}

float
pose_gradient::get_cost(const pose& _se2, cell_vector::const_iterator _begin,
                        cell_vector::const_iterator _end, jacobian* _J,
                        hessian* _H) const {
  using rectangle_d = rectangle<double>;
  using namespace internal;

  // note: all computations are done in the cell space.
  // indedices are map (m), baselink (b) and kernel (k).
  // get the transform from map (m) to kernel (k)
  const transform_type m_to_b = to_eigen(_se2.x(), _se2.y(), _se2.z());
  const transform_type b_to_k =
      to_eigen(-data_.core.center.x(), -data_.core.center.y(), 0);
  const transform_type m_to_k = m_to_b * b_to_k;

  const rectangle_d k_kernel_bb = _to_rectangle(data_.core.cost).cast<double>();
  const rectangle_d m_kernel_bb =
      (m_to_k * k_kernel_bb).array().round().matrix();

  float sum = 0;

  // init the output values
  if (_J)
    *_J = jacobian::Zero();

  if (_H)
    *_H = hessian::Zero();

  const transform_type k_to_m = m_to_k.inverse();
  const Eigen::Array2i bounds(data_.core.cost.cols, data_.core.cost.rows);

  for (; _begin != _end; ++_begin) {
    // convert to the kernel frame
    const cell k_cell =
        (k_to_m * _begin->cast<double>()).array().round().cast<int>().matrix();

    // check if k_cell is valid
    if ((k_cell.array() < 0).any() || (k_cell.array() >= bounds).any())
      continue;

    // update our outputs
    sum += data_.core.cost.at<float>(k_cell(1), k_cell(0));

    // J and H are optional
    if (_J)
      *_J -= data_.J.get(k_cell(1), k_cell(0));

    if (_H)
      *_H -= data_.H.get(k_cell(1), k_cell(0));
  }

  // flip the derivate back to the original frame.
  // note: we don't do this for the "theta"-part
  Eigen::Matrix3d rot = m_to_k.matrix();
  rot(0, 2) = 0;
  rot(1, 2) = 0;

  if (_J)
    *_J = rot * *_J;

  if (_H) {
    hessian h1 = *_H;
    hessian h2 = _H->transpose();
    *_H = (h1 + h2) * 0.5;
    *_H = rot.transpose() * *_H * rot;
  }

  return sum;
}

rectangle<int>
pose_gradient::get_bounding_box() const {
  const rectangle<int> box = internal::_to_rectangle(data_.core.cost);
  const cell origin(data_.core.center.x(), data_.core.center.y());
  return rectangle<int>{box.colwise() - origin};
}

}  // namespace dpose_core
