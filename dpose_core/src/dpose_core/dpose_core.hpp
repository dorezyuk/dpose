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

/**
 * @brief class for tolerance checks.
 *
 * Given the points _a and _b, this class will check if _a is within a
 * tolerance of _b. The "within" can be either within a sphere or within a box.
 *
 * Pass one of the modes (tolerance::mode) to the c'tor, in order to determine
 * the "within" condition.
 *
 * @code{cpp}
 * // include this header
 * #include <gpose_core/dpose_core.hpp>
 *
 * // create a tolerance of a box mode with the size 4x6
 * tolerance t(tolerance::mode::BOX, {2, 3})
 *
 * // now check is two points are within the tolerance
 * std::cout << t.within({10, 12}, {11, 9}) << std::endl; // true
 * std::cout << t.within({10, 12}, {13, 9}) << std::endl; // false
 * @endcode
 *
 */
struct tolerance {
  /// @brief diffent "modes"
  enum class mode { NONE, ANGLE, SPHERE, BOX };

  using pose = Eigen::Vector3d;

  /**
   * @brief noop-tolerance.
   *
   * All other tolerance concepts will extend this class.
   * We will use dynmic polymorphism in order to implement different concepts.
   */
  struct none_tolerance {
    virtual ~none_tolerance() = default;

    inline virtual bool
    within(const pose& _a __attribute__((unused)),
           const pose& _b __attribute__((unused))) const noexcept {
      return true;
    }
  };

  /**
   * @brief angle_tolerance
   *
   * interprets the last value of pose as a angle (in rads) and checks if the
   * angular distance from _a to _b is within the tolerance.
   */
  struct angle_tolerance : public none_tolerance {
    explicit angle_tolerance(double _tol);

    inline bool
    within(const pose& _a, const pose& _b) const noexcept {
      return std::abs(angles::shortest_angular_distance(_a.z(), _b.z())) < tol_;
    }

  private:
    double tol_;  ///< tolerance in radians
  };

  /**
   * @brief tolerance on a sphere
   *
   * The point _a is within the tolerance to _b, if their normed difference is
   * below the radius_ parameter.
   *
   * @note: _rad will be redrived as pose.norm() from the tolerance::tolerance
   * call.
   */
  struct sphere_tolerance : public none_tolerance {
    explicit sphere_tolerance(double _rad);

    inline bool
    within(const pose& _a, const pose& _b) const noexcept final {
      return (_a - _b).norm() <= radius_;
    }

  private:
    double radius_;  ///< radius
  };

  /**
   * @brief tolerance on a box
   *
   * The point _a is within the tolerance of _b if _b fits in a box of the size
   * box_ * 2 centered around _a.
   */
  struct box_tolerance : public none_tolerance {
    explicit box_tolerance(double size);
    explicit box_tolerance(const pose& _pose);

    inline bool
    within(const pose& _a, const pose& _b) const noexcept final {
      return (((_a - _b).array().abs() - box_.array()) <= 0).all();
    }

  private:
    pose box_;  ///< half size of the box
  };

private:
  using tolerance_ptr = std::unique_ptr<none_tolerance>;
  std::vector<tolerance_ptr> impl_;

  static tolerance_ptr
  factory(const mode& _m, const pose& _center) noexcept;

public:
  tolerance() = default;
  tolerance(const mode& _m, const pose& _center);

  // define the list-type. for now we dont want to use the
  // std::initializer_list, since we want to be able to dynamically allocate the
  // list.
  using pair_type = std::pair<mode, pose>;
  using list_type = std::vector<pair_type>;
  tolerance(const list_type& _list);

  inline bool
  within(const pose& _a, const pose& _b) const noexcept {
    // check all tolerances defined
    return std::all_of(
        impl_.begin(), impl_.end(),
        [&](const tolerance_ptr& _impl) { return _impl->within(_a, _b); });
  }
};

/**
 * @brief gradient decent optimizer for the pose_gradient.
 *
 * Will perform the gradient decent until a termination condition is met.
 * The decent ends
 * - if either the maximum iterations are reached,
 * - if the cost lies below the epsilon bound,
 * - if the derived position lies outside of the tolerance tol w.r.t _start.
 */
struct gradient_decent {
  /// @brief parameter for the optimization
  struct parameter {
    size_t iter = 10;     ///< maximal number of steps
    double step_t = 1;     ///< maximal step size for translation (in cells)
    double step_r = 0.1;   ///< maximal step size for rotation (in rads)
    double epsilon = 0.1;  ///< cost-threshold for termination
    tolerance tol;        ///< maximal distance from _start
  };

  gradient_decent() = default;
  explicit gradient_decent(parameter&& _param) noexcept;

  std::pair<float, Eigen::Vector3d>
  solve(const pose_gradient& _pg, const Eigen::Vector3d& _start) const;

private:
  parameter param_;  ///< parameterization for the optimization
};

}  // namespace dpose_core

#endif  // DPOSE_CORE__DPOSE_CORE__HPP