#ifndef DPOSE_CORE__DPOSE_CORE__HPP
#define DPOSE_CORE__DPOSE_CORE__HPP

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

/// @brief POD holding all the data required for optimization
struct data {
  cv::Mat cost;     ///< cost matrix
  cv::Mat d_x;      ///< derivative of the cost in x
  cv::Mat d_y;      ///< derivative of the cost in y
  cv::Mat d_theta;  ///< derivative of the cost in theta

  Eigen::Vector2i center;  ///< center cell
};

/// @brief POD defining the parameters
struct parameter {
  unsigned int padding = 2;  ///< padding of the given footprint (in cells)
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
 * const auto res = pg.get_cost(my_pose);
 * @endcode
 *
 * You can use this class for your own optimization, or reuse the
 * gradient-decent solver below.
 *
 * The output is in the global frame.
 */
struct pose_gradient {
  using parameter = internal::parameter;

  pose_gradient() = default;
  pose_gradient(costmap_2d::Costmap2D& _cm, const polygon_msg& _footprint,
                const parameter& _param);
  pose_gradient(costmap_2d::LayeredCostmap& _lcm, const parameter& _param);

  std::pair<float, Eigen::Vector3d>
  get_cost(const Eigen::Vector3d& _se2) const;

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
 */
struct tolerance {
  /// @brief diffent "modes"
  enum class mode { NONE, SPHERE, BOX };

  using pose = Eigen::Vector2d;

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
   * @brief tolerance on a sphere
   *
   * The point _a is within the tolerance to _b, if their normed difference is
   * below the rad_ parameter.
   *
   * @note: _rad will be redrived as pose.norm() from the tolerance::tolerance
   * call.
   */
  struct sphere_tolerance : public none_tolerance {
    explicit sphere_tolerance(double _rad);

    inline bool
    within(const pose& _a, const pose& _b) const noexcept final {
      return (_a - _b).norm() <= rad_;
    }

  private:
    double rad_;  ///< radius
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

  tolerance();
  tolerance(const mode& _m, const pose& _center);

  inline bool
  within(const pose& _a, const pose& _b) const noexcept {
    return impl_->within(_a, _b);
  }

private:
  std::unique_ptr<none_tolerance> impl_;
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
    size_t iter = 10;      ///< maximal number of steps
    double step_t = 1;     ///< maximal step size for translation (in cells)
    double step_r = 0.1;   ///< maximal step size for rotation (in rads)
    double epsilon = 0.1;  ///< cost-threshold for termination
    tolerance tol;         ///< maximal distance from _start
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