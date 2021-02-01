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

#include <vector>

namespace dpose_core {

/// @brief a closed rectangle (hence 5 columns)
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

/// @brief polygon where first row holds the x, and second row y values.
using polygon = Eigen::Matrix<int, 2UL, Eigen::Dynamic>;

// todo add alignment
using cell = Eigen::Vector2i;
using cell_vector = std::vector<cell>;

// forward-decleration so we can befriend these stuctures together.
struct jacobian_data;
struct hessian_data;

/// @brief struct holding the "core" data - just cost matrix and center cell
struct cost_data {
  // both need access to the raw data
  friend jacobian_data;
  friend hessian_data;

  cost_data() = default;
  cost_data(const polygon& _footprint, size_t _padding);

  /// @brief returns the cost at the given x and y.
  /// @param _x column of the pixel.
  /// @param _y row 0f the pixel.
  inline double
  at(int _y, int _x) const {
    return static_cast<double>(cost.at<float>(_y, _x));
  }

  /// @brief returns the cost at the given pixel.
  /// @param _cell coordinate of the pixel.
  inline double
  at(const cell& _cell) const {
    return at(_cell.y(), _cell.x());
  }

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
 * @brief struct holding the data required for the jacobians.
 *
 * Use jacobian_data::at to get the jacobian for a pixel.
 * The Jacobian has the form \f$ [f_x, f_y, f_{\theta}]^T \f$, where the
 * \f$ f_x \f$ represents the partial derivative of the cost f with respect to
 * x.
 */
struct jacobian_data {
  // the Hessian is build on top of the Jacobian
  friend hessian_data;

  jacobian_data() = default;
  explicit jacobian_data(const cost_data& _cost);

  /// @brief the data-structure for the user.
  using jacobian = Eigen::Vector3d;

  /// @brief returns the jacobian for given x and y.
  /// @param _x column of the pixel.
  /// @param _y row 0f the pixel.
  inline jacobian
  at(int _y, int _x) const {
    return {static_cast<double>(d_x.at<float>(_y, _x)),
            static_cast<double>(d_y.at<float>(_y, _x)),
            static_cast<double>(d_z.at<float>(_y, _x))};
  }

  /// @brief returns the jacobian at the given pixel.
  /// @param _cell coordinate of the pixel.
  inline jacobian
  at(const cell& _cell) const {
    return at(_cell.y(), _cell.x());
  }

  inline double
  at(size_t _z, int _y, int _x) const {
    switch (_z) {
      case 0: return static_cast<double>(d_x.at<float>(_y, _x)); break;
      case 1: return static_cast<double>(d_y.at<float>(_y, _x)); break;
      case 2: return static_cast<double>(d_z.at<float>(_y, _x)); break;
      default: throw std::out_of_range("invalid z index");
    }
  }

private:
  cv::Mat d_x;  ///< derivative of the cost in x
  cv::Mat d_y;  ///< derivative of the cost in y
  cv::Mat d_z;  ///< derivative of the cost in z (theta)
};

/**
 * @brief struct holding the data required for the hessians.
 *
 * Use hessian_data::get to get the hessian for a pixel.
 * The Hessian has the form
 * \f[
 * \begin{bmatrix}
 * f_{x, x} & f_{x, y} & f_{x, \theta} \\
 * f_{y, x} & f_{y, y} & f_{y, \theta} \\
 * f_{\theta, x} & f_{theta, y} & f_{\theta \theta}
 * \end{bmatrix}
 * \f]
 *
 * \f$ f_{x y} \f$ indicates the partial derivative of the cost f with respect
 * to x and y. The Hessian is symmetric.
 */
struct hessian_data {
  hessian_data() = default;
  hessian_data(const cost_data& _cost, const jacobian_data& _jacobian);

  /// @brief the data-structure for the user.
  using hessian = Eigen::Matrix3d;

  /// @brief returns the hessian for given x and y.
  /// @param _x column of the pixel.
  /// @param _y row 0f the pixel.
  inline hessian
  at(int _y, int _x) const {
    // todo check if this allocation hurts us...
    hessian H;
    // clang-format off
    H << d_x_x.at<float>(_y, _x), d_y_x.at<float>(_y, _x), d_z_x.at<float>(_y, _x),
         d_y_x.at<float>(_y, _x), d_y_y.at<float>(_y, _x), d_y_z.at<float>(_y, _x),
         d_z_x.at<float>(_y, _x), d_y_z.at<float>(_y, _x), d_z_z.at<float>(_y, _x);
    // clang-format on
    return H;
  }

  /// @brief returns the hessian at the given pixel.
  /// @param _cell coordinate of the pixel.
  inline hessian
  at(const cell& _cell) const {
    return at(_cell.y(), _cell.x());
  }

private:
  cv::Mat d_x_x;  ///< derivative of the cost in x,x
  cv::Mat d_y_x;  ///< derivative of the cost in y,x
  cv::Mat d_z_x;  ///< derivative of the cost in theta,x
  cv::Mat d_y_y;  ///< derivative of the cost in y,y
  cv::Mat d_y_z;  ///< derivative of the cost in y,theta
  cv::Mat d_z_z;  ///< derivative of the cost in theta,theta
};

/// @brief POD holding all the data required for optimization
/// This POD is owned by the pose_gradient (see below)
struct data {
  cost_data core;   ///< hold the data related to the costs
  jacobian_data J;  ///< holds the data related to the Jacobian
  hessian_data H;   ///< holds the data related to the Hessian
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
    bool generate_hessian = false;
  };

  // the three arguments for get_pose
  using jacobian = jacobian_data::jacobian;
  using hessian = hessian_data::hessian;
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
  /// @param[out] _H optional hessian. will be ignored if nullptr
  float
  get_cost(const pose& _se2, cell_vector::const_iterator _begin,
           cell_vector::const_iterator _end, jacobian* _J, hessian* _H) const;

  inline const data&
  get_data() const noexcept {
    return data_;
  }

private:
  data data_;
};

}  // namespace dpose_core

#endif  // DPOSE_CORE__DPOSE_CORE__HPP