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

/// @brief POD holding all the data required for operation
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

using polygon = Eigen::Matrix<int, 2ul, Eigen::Dynamic>;
using cell_type = cv::Point2i;
using cell_vector_type = std::vector<cell_type>;


/**
 * @brief Returns an image with a polygon defined by _cells
 *
 * The image will have the size of the bounding box of _cells.
 * The drawn polygon will be closed and the value of the line is 255.
 * The function will shift the origin such that the smallest cell in _cells is
 * [0, 0]. The origin is then moved by new_origin = old_origin - _shift.
 *
 * @param[in] _cells an polygon
 * @param[out] _shift shifted origin of the _cells
 * @param[in] _padding optional padding to each side. padding cannot be
 * negative.
 *
 * @return cv::Mat image showing the polygon
 * @throw std::invalid_argument if _cells is empty or if padding is negative
 */
cv::Mat
draw_polygon(const cell_vector_type& _cells, cell_type& _shift,
             const cell_type& _padding = cell_type(0, 0));

/**
 * @brief Post-processing function.
 *
 * We will add a small gaussian blur at the edges of the polygon, so the
 * derivative is a little friendlier for the optimizers.
 *
 * @param _image original image
 * @param _cells outline of the polygon
 * @return cv::Mat smoothed image
 */
cv::Mat
smoothen_edges(cv::InputArray _image, const cell_vector_type& _cells);

/**
 * @brief Returns cells laying on the perimeter of a circle.
 *
 * Function implements the circular bresenham algorithm. It returns all cells
 * at a circle defined by the _center and _radius. Those cells are sorted CW,
 * and without duplicates.
 *
 * @param _center center of the circle
 * @param _radius radius of the circle
 * @return cells at the perimeter of the circle
 */
cell_vector_type
get_circular_cells(const cell_type& _center, size_t _radius) noexcept;

/**
 * @brief Computes the circular derivative around the _center.
 *
 * The derivative for a cell is approximated by the difference if its neighbor
 * cells, laying in the same circle (around the _center cell).
 *
 * @param _image input for the derivative calculation
 * @param _center center of rotation
 * @return matrix with the angular derivate of _image around _center.
 */
cv::Mat
angular_derivative(cv::InputArray _image, const cell_type& _center);


data
init_data(const polygon& _footprint, const parameter& _param);

// cell_vector_type
// to_cells(const polygon_msg& _footprint, double _resolution);

}  // namespace internal

namespace gm = geometry_msgs;
namespace cm = costmap_2d;
namespace eg = Eigen;

using point_msg = gm::Point;
using pose_msg = gm::Pose;
using polygon_msg = std::vector<gm::Point>;

using transform_type = Eigen::Isometry2d;

inline transform_type
to_eigen(double _x, double _y, double _yaw) noexcept {
  return Eigen::Translation2d(_x, _y) * Eigen::Rotation2Dd(_yaw);
}




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