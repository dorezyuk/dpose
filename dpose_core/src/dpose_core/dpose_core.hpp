#ifndef DPOSE_CORE__DPOSE_CORE__HPP
#define DPOSE_CORE__DPOSE_CORE__HPP

#include <costmap_2d/costmap_2d.h>
#include <costmap_2d/layered_costmap.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <Eigen/Dense>

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
data
init_data(const polygon& _footprint, const parameter& _param);

}  // namespace internal

namespace gm = geometry_msgs;
namespace cm = costmap_2d;
namespace eg = Eigen;

using point_msg = gm::Point;
using pose_msg = gm::Pose;
using polygon_msg = std::vector<gm::Point>;


struct pose_gradient {
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

struct gradient_decent {
  struct parameter {
    size_t iter;
    double step_t;
    double step_r;
    double epsilon;
  };

  static std::pair<float, Eigen::Vector3d>
  solve(const pose_gradient& _laces, const Eigen::Vector3d& _start,
        const parameter& _param);
};

}  // namespace dpose_core

#endif  // DPOSE_CORE__DPOSE_CORE__HPP