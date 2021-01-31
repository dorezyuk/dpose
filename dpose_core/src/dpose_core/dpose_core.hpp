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

#include <angles/angles.h>
#include <costmap_2d/costmap_2d.h>
#include <costmap_2d/layered_costmap.h>
#include <geometry_msgs/Point.h>

#include <opencv2/opencv.hpp>

#include <Eigen/Dense>

#include <memory>
#include <utility>
#include <vector>

namespace dpose_core {
namespace internal {

/// @brief struct holding the "core" data - just cost matrix and center cell
struct cost_data {
  cv::Mat cost;            ///< cost matrix
  Eigen::Vector2i center;  ///< center cell
};

/// @brief struct holding the data required for the jacobians
/// Use jacobian_data::get to get the jacobian for a pixel.
struct jacobian_data {
  /*
   * the jacobian has the form
   * [x, y, z]^T
   */
  using jacobian = Eigen::Vector3d;

  /// @brief returns the jacobian for given x and y.
  inline jacobian
  get(int _y, int _x) const {
    return {static_cast<double>(d_x.at<float>(_y, _x)),
            static_cast<double>(d_y.at<float>(_y, _x)),
            static_cast<double>(d_z.at<float>(_y, _x))};
  }

  cv::Mat d_x;  ///< derivative of the cost in x
  cv::Mat d_y;  ///< derivative of the cost in y
  cv::Mat d_z;  ///< derivative of the cost in z (theta)
};

/// @brief struct holding the data required for the hessians
/// Use hessian_data::get to get the hessian for a pixel
struct hessian_data {
  /*
   * the hessian has the form
   * [[x_x, x_y, x_z],
   *  [y_x, y_y, y_z],
   *  [z_x, z_y, z_z]]
   */
  using hessian = Eigen::Matrix3d;

  /// @brief returns the hessian for given x and y.
  inline hessian
  get(int _y, int _x) const {
    hessian H;
    // clang-format off
    H << d_x_x.at<float>(_y, _x), d_x_y.at<float>(_y, _x), d_x_z.at<float>(_y, _x),
         d_y_x.at<float>(_y, _x), d_y_y.at<float>(_y, _x), d_y_z.at<float>(_y, _x),
         d_z_x.at<float>(_y, _x), d_z_y.at<float>(_y, _x), d_z_z.at<float>(_y, _x);
    // clang-format on
    return H;
  }

  cv::Mat d_x_x;  ///< derivative of the cost in x,x
  cv::Mat d_x_y;  ///< derivative of the cost in x,y
  cv::Mat d_x_z;  ///< derivative of the cost in x,theta
  cv::Mat d_y_x;  ///< derivative of the cost in y,x
  cv::Mat d_y_y;  ///< derivative of the cost in y,y
  cv::Mat d_y_z;  ///< derivative of the cost in y,theta
  cv::Mat d_z_x;  ///< derivative of the cost in theta,x
  cv::Mat d_z_y;  ///< derivative of the cost in theta,y
  cv::Mat d_z_z;  ///< derivative of the cost in theta,theta
};

/// @brief POD holding all the data required for optimization
struct data {
  cost_data core;
  jacobian_data J;
  hessian_data H;
};

/// @brief POD defining the parameters
struct parameter {
  unsigned int padding = 2;  ///< padding of the given footprint (in cells)
  bool generate_hessian = false;
};

/// @brief polygon where first row holds the x, and second row y values.
using polygon = Eigen::Matrix<int, 2UL, Eigen::Dynamic>;

/// @brief constructs cost and its derivatives from the inputs
/// @param _footprint the footprint (may or may not be closed)
/// @param _param parameters for the operation
/// @throws std::runtime_error if the _footprint is ill-formed
/// @note will safe some images to "/tmp" if compiled as assert enabled
data
init_data(const polygon& _footprint, const parameter& _param);

}  // namespace internal

namespace cm = costmap_2d;

/// @brief ros-specific polygon definition
using polygon_msg = std::vector<geometry_msgs::Point>;

/**
 * @brief computes a pose-gradient from the given costmap
 *
 * This is most-likely the entry-point for this lib.
 * Use pose_gradient::get_cost to obtain the cost and the gradient.
 * The input pose is must be in the global frame of the provided costmap.
 *
 * @code{cpp}
 * // include this header
 * #include <dpose_core/dpose_core.hpp>
 *
 * // construct an instance from your costmap and the footprint
 * dpose_core::pose_gradient pg(my_costmap, my_footprint);
 *
 * // get the gradient for a pose
 * Eigen::Vector3d grad;
 * const auto res = pg.get_cost(my_pose, grad, nullptr);
 * @endcode
 *
 * You can use this class for your own optimization, or reuse the
 * gradient-decent solver below.
 *
 * The output is in the global frame.
 * The input (and output) is in cell-domain (not metric).
 */
struct pose_gradient {
  using parameter = internal::parameter;

  // the three arguments for get_pose
  using jacobian = internal::jacobian_data::jacobian;
  using hessian = internal::hessian_data::hessian;
  using pose = Eigen::Vector3d;

  pose_gradient() = default;
  pose_gradient(costmap_2d::Costmap2D& _cm, const polygon_msg& _footprint,
                const parameter& _param);
  pose_gradient(costmap_2d::LayeredCostmap& _lcm, const parameter& _param);

  /// @brief returns the cost for the given se2 pose
  /// @param[in] _se2 pose of interest. should be in the global frame.
  /// @param[out] _J optional jacobian. will be ignored if nullptr
  /// @param[out] _H optional hessian. will be ignored if nullptr
  float
  get_cost(const pose& _se2, jacobian* _J, hessian* _H) const;

private:
  internal::data data_;
  // promise not to alter the costmap, but this class does not have a
  // const-correctness concept
  mutable costmap_2d::Costmap2D* cm_ = nullptr;
};

}  // namespace dpose_core

#endif  // DPOSE_CORE__DPOSE_CORE__HPP