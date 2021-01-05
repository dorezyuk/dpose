#ifndef DPOSE_CORE__DPOSE_CORE__HPP
#define DPOSE_CORE__DPOSE_CORE__HPP

#include <costmap_2d/costmap_2d.h>
#include <costmap_2d/layered_costmap.h>
#include <geometry_msgs/Point.h>

#include <opencv2/opencv.hpp>

#include <Eigen/Dense>

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
  unsigned int padding = 2;  ///< padding of the given footprint. setting
};

/// @brief polygon where first row holds the x, and second row y values.
using polygon = Eigen::Matrix<int, 2ul, Eigen::Dynamic>;

/// @brief constructs cost and its derivatives from the inputs
/// @param _footprint the footprint (may or may not be closed)
/// @param _param parameters for the operation
/// @throws std::runtime_error if the _footprint is ill-formed
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
  pose_gradient(costmap_2d::Costmap2D& _cm, const polygon_msg& _footprint);
  explicit pose_gradient(costmap_2d::LayeredCostmap& _lcm);

  std::pair<float, Eigen::Vector3d>
  get_cost(const Eigen::Vector3d& _se2) const;

private:
  internal::data data_;
  // promise not to alter the costmap, but this class does not have a
  // const-correctness concept
  mutable costmap_2d::Costmap2D* cm_ = nullptr;
};

/**
 * @brief gradient decent optimizer for the pose_gradient.
 *
 * Will perform the gradient decent until a termination condition is met.
 * The decent ends, if either the maximum iterations are reached or if the cost
 * lies below the epsilon bound.
 */
struct gradient_decent {
  /// @brief parameter for the optimization
  struct parameter {
    size_t iter = 10;      ///< maximal number of steps
    double step_t = 1;     ///< maximal step size for translation (in cells)
    double step_r = 0.1;   ///< maximal step size for rotation (in rads)
    double epsilon = 0.1;  ///< cost-threshold for termination
  };

  gradient_decent() = default;
  gradient_decent(const parameter& _param) noexcept;
  gradient_decent(parameter&& _param) noexcept;

  std::pair<float, Eigen::Vector3d>
  solve(const pose_gradient& _pg, const Eigen::Vector3d& _start) const;

private:
  parameter param_;  ///< parameterization for the optimization
};

}  // namespace dpose_core

#endif  // DPOSE_CORE__DPOSE_CORE__HPP