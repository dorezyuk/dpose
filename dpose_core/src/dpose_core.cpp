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

/// @brief open-cv specific data-types
using _cv_cell = cv::Point2i;
using _cv_cell_vector = std::vector<_cv_cell>;

/// @brief converts eigen to open-cv data
_cv_cell_vector
_to_open_cv(const polygon& _footprint) noexcept {
  _cv_cell_vector cells;
  cells.reserve(_footprint.cols());
  for (int cc = 0; cc != _footprint.cols(); ++cc)
    cells.emplace_back(_footprint(0, cc), _footprint(1, cc));

  return cells;
}

/// @brief computes a cost-image based on the polygon
/// @param _cells corners of a sparse polygon
/// @param _padding additional padding
cv::Mat
_get_cost(const polygon& _fp, size_t& _padding) {
  // convert the eigen-polygon to opencv-cells
  const auto _cells = _to_open_cv(_fp);

  // get the bounding box
  const cv::Rect bb = cv::boundingRect(_cells);

  // we need distinct points
  assert(!bb.empty() && "bounding box cannot be empty");

  // apply our padding
  cv::Size bb_size = bb.size();
  bb_size.width += (_padding * 2);
  bb_size.height += (_padding * 2);

  // setup the image and draw the cells
  cv::Mat image(bb_size, cv::DataType<uint8_t>::type, cv::Scalar(255));
  cv::polylines(image, _cells, true, cv::Scalar(0));

  // image cannot be just "white"
  assert(cv::imwrite("/tmp/poly.jpg", image));
  assert(cv::countNonZero(image) != image.cols * image.rows &&
         "no polygon drawn");

  // get the euclidean distance transform
  cv::Mat edt(bb_size, cv::DataType<float>::type);
  cv::distanceTransform(image, edt, cv::DIST_L2, cv::DIST_MASK_PRECISE,
                        cv::DataType<float>::type);

  assert(cv::imwrite("/tmp/edt.jpg", edt));
  assert(cv::countNonZero(edt) > 0 && "distance transform failed");

  // we now tweak the costs out side of the polygon (outer), such that we have a
  // nice cost-function.

  // get the mask of the polygon defined by cells
  image.setTo(cv::Scalar(255));
  std::vector<_cv_cell_vector> cells({_cells});
  cv::fillPoly(image, cells, cv::Scalar(0));

  assert(cv::countNonZero(image) > 0 && "filling of the mask failed");

  // scale the outer part down, since in the context of obstacle avoidance, the
  // outer part is less bad...
  cv::Mat outer(bb_size, cv::DataType<float>::type, cv::Scalar(0));
  cv::copyTo(edt, outer, image);

  // todo make this a parameter
  outer *= -0.01;
  double min;
  cv::minMaxIdx(outer, &min);

  // copy the distance transform within the mask into final image
  cv::copyTo(outer, edt, image);
  edt -= min;
  cv::GaussianBlur(edt, edt, cv::Size(3, 3), 0);

  return edt;
}

/// @brief generates a rectangle with the size of _cm
inline rectangle<int>
_to_rectangle(const cv::Mat& _cm) noexcept {
  return to_rectangle<int>(_cm.cols, _cm.rows);
}

cost_data::cost_data(const polygon& _footprint, size_t _padding) {
  // get the center
  const cell padding(_padding, _padding);
  center = padding - _footprint.rowwise().minCoeff();

  // shift everything by the center, so we just have positive values
  polygon footprint = _footprint.colwise() + center;
  assert(footprint.array().minCoeff() == static_cast<int>(_padding) &&
         "footprint shifting failed");

  // get the cost image
  cost = _get_cost(footprint, _padding);
  // safe the image if we are running in debug mode (and scale the images)
  assert(cv::imwrite("/tmp/cost.jpg", cost * 100));

  // create the bounding box around the original footprint
  box = _to_rectangle(cost).colwise() - center;
}

using transform_type = Eigen::Isometry2d;

inline transform_type
to_eigen(double _x, double _y, double _yaw) noexcept {
  return Eigen::Translation2d(_x, _y) * Eigen::Rotation2Dd(_yaw);
}

pose_gradient::pose_gradient(const polygon& _footprint,
                             const parameter& _param) {
  // we need an area
  // todo maybe add a check if the footprint defines a real area...
  if (_footprint.cols() < 3)
    throw std::runtime_error("footprint must contain at least three points");

  data_ = cost_data(_footprint, _param.padding);
}

inline void
_interpolate(const cv::Mat& _image, const Eigen::Array2i& _lower,
             const Eigen::Array2i _upper, Eigen::Matrix2d& _m) {
  // m is [[f_00, f_01], [f_10, f_11]]
  _m << _image.at<float>(_lower(1), _lower(0)),
      _image.at<float>(_upper(1), _lower(0)),
      _image.at<float>(_lower(1), _upper(0)),
      _image.at<float>(_upper(1), _upper(0));
}

interpolator::interpolator(const cv::Mat& _image) :
    image_(_image), bounds(_image.cols - 1, _image.rows - 1) {}

bool
interpolator::init(const Eigen::Array2d& k_cell) {
  // interpolate the cost: get the cell-indices of interest.
  // see https://en.wikipedia.org/wiki/Bilinear_interpolation for details.
  // todo if i switch to ceil and make the lower left cornor the anchor, it's
  // gonna be faster
  k_upper = k_cell.round().cast<int>();
  k_lower = k_upper - 1;

  // check if the end-points are valid
  if ((k_lower < 1).any() || (k_upper >= bounds.cast<int>()).any())
    return false;

  // init the m-matrix
  _interpolate(image_, k_lower, k_upper, m);
  return true;
}

double
interpolator::get(const Eigen::Array2d& k_cell) {
  // c_rel is the normalized point w.r.t a cell.
  // c_rel is defined in [0, 1]^2
  c_rel = k_cell.array() - k_upper.cast<double>() + 0.5;
  c_rel_x << 1 - c_rel(0), c_rel(0);
  c_rel_y << 1 - c_rel(1), c_rel(1);

  return c_rel_x.transpose() * m * c_rel_y;
}

double
pose_gradient::get_cost(const pose& _se2, cell_vector::const_iterator _begin,
                        cell_vector::const_iterator _end, jacobian* _J) const {
  // note: all computations are done in the cell space.
  // indedices are map (m), baselink (b) and kernel (k).
  // get the transform from map (m) to kernel (k)
  const auto& center = data_.get_center();
  const transform_type m_to_b = to_eigen(_se2.x(), _se2.y(), _se2.z());
  const transform_type b_to_k = to_eigen(-center.x(), -center.y(), 0);
  const transform_type m_to_k = m_to_b * b_to_k;

  constexpr double offset = 1e-6;

  // clang-format off
  const transform_type x_lower = (to_eigen(_se2.x() - offset, _se2.y(), _se2.z()) * b_to_k).inverse();
  const transform_type x_upper = (to_eigen(_se2.x() + offset, _se2.y(), _se2.z()) * b_to_k).inverse();
  const transform_type y_lower = (to_eigen(_se2.x(), _se2.y() - offset, _se2.z()) * b_to_k).inverse();
  const transform_type y_upper = (to_eigen(_se2.x(), _se2.y() + offset, _se2.z()) * b_to_k).inverse();
  const transform_type z_lower = (to_eigen(_se2.x(), _se2.y(), _se2.z() - offset) * b_to_k).inverse();
  const transform_type z_upper = (to_eigen(_se2.x(), _se2.y(), _se2.z() + offset) * b_to_k).inverse();
  // clang-format on
  double sum = 0;

  // init the output values
  if (_J)
    *_J = jacobian::Zero();

  const transform_type k_to_m = m_to_k.inverse();

  Eigen::Array2d k_cell;
  interpolator ip(data_.get_data());

  for (; _begin != _end; ++_begin) {
    // convert to the kernel frame
    k_cell = (k_to_m * _begin->cast<double>()).array();

    // returns false if the k_cell is outsize of the kernel-bounds
    if (!ip.init(k_cell))
      continue;

    // update the cost
    sum += ip.get(k_cell);

    // J is optional
    if (_J) {
      const double dx_lower = ip.get(x_lower * _begin->cast<double>().array());
      const double dx_upper = ip.get(x_upper * _begin->cast<double>().array());
      _J->x() += ((dx_upper - dx_lower) * 5e5);

      const double dy_lower = ip.get(y_lower * _begin->cast<double>().array());
      const double dy_upper = ip.get(y_upper * _begin->cast<double>().array());
      _J->y() += ((dy_upper - dy_lower) * 5e5);

      const double dz_lower = ip.get(z_lower * _begin->cast<double>().array());
      const double dz_upper = ip.get(z_upper * _begin->cast<double>().array());
      _J->z() += ((dz_upper - dz_lower) * 5e5);
    }
  }

  return sum;
}

}  // namespace dpose_core
