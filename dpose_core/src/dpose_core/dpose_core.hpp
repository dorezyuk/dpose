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
#ifndef DPOSE_CORE__DPOSE_CORE__HPP
#define DPOSE_CORE__DPOSE_CORE__HPP

#include <opencv2/opencv.hpp>

#include <Eigen/Dense>

#include <array>
#include <stdexcept>
#include <vector>

namespace dpose_core {

/// @brief a closed rectangle (hence 5 columns)
/// First row holds the x, and second row y values.
template <typename _T>
using rectangle = Eigen::Matrix<_T, 2, 5>;

/// @brief constructs a rectangle with the given width and height
/// @param _w width of the rectangle
/// @param _h height of the rectangle
template <typename _T>
inline rectangle<_T>
to_rectangle(const _T& w, const _T& h) noexcept {
  rectangle<_T> box;
  // clang-format off
  box << 0, w, w, 0, 0,
         0, 0, h, h, 0;
  // clang-format on
  return box;
}

/// @brief constructs a rectangle with the given size
/// @param _min lower corner
/// @param _max upper corner
template <typename _T>
inline rectangle<_T>
to_rectangle(const Eigen::Matrix<_T, 2, 1>& _min,
             const Eigen::Matrix<_T, 2, 1>& _max) noexcept {
  rectangle<_T> box;
  // clang-format off
  box << _min.x(), _max.x(), _max.x(), _min.x(), _min.x(),
         _min.y(), _min.y(), _max.y(), _max.y(), _min.y();
  // clang-format on
  return box;
}

/// @brief polygon where first row holds the x, and second row y values.
using polygon = Eigen::Matrix<int, 2UL, Eigen::Dynamic>;

// todo add alignment
using cell = Eigen::Vector2i;
using cell_vector = std::vector<cell>;

/// @brief struct holding the "core" data - just cost matrix and center cell
struct cost_data {
  cost_data() = default;
  cost_data(const polygon& _footprint, size_t _padding);

  inline const cv::Mat&
  get_data() const noexcept {
    return cost;
  }

  inline const cell&
  get_center() const noexcept {
    return center;
  }

  inline const rectangle<int>&
  get_box() const noexcept {
    return box;
  }

private:
  cv::Mat cost;        ///< cost matrix
  cell center;         ///< center cell
  rectangle<int> box;  ///< the bounding box
};

/**
 * @brief Implements bilinear interpolation
 *
 * Call first init() to fetch the cells of interest. Call then get() to the get
 * interpolated values. The functions are seperated to speedup the gradient
 * computation which uses the same cell-data.
 */
struct interpolator {
  interpolator(const cv::Mat& _image);

  bool
  init(const Eigen::Array2d& k_cell);

  double
  get(const Eigen::Array2d& k_cell);

private:
  const Eigen::Array2d bounds;
  const cv::Mat image_;
  Eigen::Array2i k_lower, k_upper;
  Eigen::Vector2d c_rel, c_rel_x, c_rel_y;
  Eigen::Matrix2d m;
};

/**
 * @brief computes a pose-gradient from the given costmap
 *
 * The input (and output) is in cell-domain (not metric).
 */
struct pose_gradient {
  /// @brief POD defining the parameters
  struct parameter {
    unsigned int padding = 2;  ///< padding of the given footprint (in cells)
  };

  // the three arguments for get_pose
  using jacobian = Eigen::Vector3d;
  using pose = Eigen::Vector3d;

  /// @brief sets up internal data based on _footprint and _param.
  /// @param _footprint the footprint (may or may not be closed)
  /// @param _param parameters for the operation
  /// @throws std::runtime_error if the _footprint is ill-formed
  /// @note will safe some images to "/tmp" if compiled as assert enabled
  pose_gradient(const polygon& _footprint, const parameter& _param);
  pose_gradient() = default;

  /// @brief returns the cost for the given se2 pose
  /// @param[in] _begin begin of a cell-vector with lethal costs
  /// @param[in] _end end of a cell-vector with lethal costs
  /// @param[in] _se2 pose of interest. should be in the global frame.
  /// @param[out] _J optional jacobian. will be ignored if nullptr
  double
  get_cost(const pose& _se2, cell_vector::const_iterator _begin,
           cell_vector::const_iterator _end, jacobian* _J) const;

  inline const cost_data&
  get_data() const noexcept {
    return data_;
  }

private:
  cost_data data_;
};

}  // namespace dpose_core

#endif  // DPOSE_CORE__DPOSE_CORE__HPP