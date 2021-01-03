#ifndef DPOSE_LAYER__DPOSE_LAYER__HPP
#define DPOSE_LAYER__DPOSE_LAYER__HPP

#include <dpose_core/dpose_core.hpp>

#include <costmap_2d::Layer>

#include <Eigen/Dense>

namespace dpose_layer {

struct DposeLayer : public costmap_2d::Layer {
  void
  updateBounds(double robot_x, double robot_y, double robot_yaw, double*,
               double*, double*, double*) override;

  void
  updateCosts(costmap_2d::Costmap2D& map, int, int, int, int) override;

  void
  onFootprintChanged() override;

protected:
  void
  onInitialize() override;

private:
  ros::Publisher d_pub_;
  Eigen::Vector3d robot_pose_;
  laces_ros impl_;
  gradient_decent::parameter param_;
};

}  // namespace dpose_layer

#endif  // DPOSE_LAYER__DPOSE_LAYER__HPP