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
#include <dpose_goal_tolerance/dpose_goal_tolerance.hpp>

#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <xmlrpcpp/XmlRpcException.h>
#include <xmlrpcpp/XmlRpcValue.h>

#include <cmath>
#include <memory>
#include <string>
#include <utility>

/// @brief the name of the module
constexpr char _name[] = "[dpose_goal_tolerance]: ";

// custom prints
#define DP_INFO_LOG(args) ROS_INFO_STREAM(_name << args)
#define DP_WARN_LOG(args) ROS_WARN_STREAM(_name << args)
#define DP_ERROR_LOG(args) ROS_ERROR_STREAM(_name << args)
#define DP_FATAL_LOG(args) ROS_FATAL_STREAM(_name << args)

namespace dpose_goal_tolerance {

namespace internal {

tolerance::angle_tolerance::angle_tolerance(double _tol) :
    tol_(angles::normalize_angle_positive(_tol)) {}

tolerance::sphere_tolerance::sphere_tolerance(const pose& _norm) :
    radius_(_norm.norm()) {}

tolerance::box_tolerance::box_tolerance(const pose& _box) :
    box_(_box.array().abs().matrix()) {}

tolerance::tolerance_ptr
tolerance::factory(const mode& _m, const pose& _p) noexcept {
  switch (_m) {
    case mode::NONE: return nullptr;
    case mode::ANGLE: return tolerance_ptr{new angle_tolerance(_p.z())};
    case mode::SPHERE: return tolerance_ptr{new sphere_tolerance(_p)};
    case mode::BOX: return tolerance_ptr{new box_tolerance(_p)};
  }
  return nullptr;
}

tolerance::tolerance(const mode& _m, const pose& _center) {
  if (_m != mode::NONE)
    impl_.emplace_back(factory(_m, _center));
}

tolerance::tolerance(const list_type& _list) {
  for (const auto& pair : _list) {
    if (pair.first != mode::NONE)
      impl_.emplace_back(factory(pair.first, pair.second));
  }
}

// short-cuts
using XmlRpc::XmlRpcException;
using XmlRpc::XmlRpcValue;

/// @brief returns a value from _v under _tag or the _default value.
/// @param _v data-storage
/// @param _tag the tag under which the data is saved
/// @param _default the default value
/// If anything goes wrong, the function will fall-back to the _default value.
template <typename _T>
_T
_getElement(const XmlRpcValue& _v, const std::string& _tag,
            const _T& _default) noexcept {
  // check if the tag is defined
  if (!_v.hasMember(_tag))
    return _default;

  // try to get the desired value
  try {
    return static_cast<_T>(_v[_tag]);
  }
  catch (const XmlRpcException& _ex) {
    return _default;
  }
}

/// @brief returns a pose from the element _v.
/// _v can have the form {"x": a, "y": b, "z": c}.
/// missing values will be replaced with 0
inline tolerance::pose
_getPose(const XmlRpcValue& _v) noexcept {
  return {_getElement(_v, "x", 0.), _getElement(_v, "y", 0.),
          _getElement(_v, "z", 0.)};
}

/// @brief builds a tolerance class from the parameters at the ros-server
/// @param _name of the parameter
/// @param _nh node-handle with the right namespace
/*
 * your yaml file might look something like
 *
 * my_parameter:
 *  - {type: "box", x: 0.1, y: 0.2, z: inf}
 *  - {type: "sphere", x: 0.5}
 *  - {type: "angular", x: 0.2}
 *
 * Then you can load it with
 * ros::NodeHandle nh("~");
 * const auto tolerance = _loadTolerance("my_parameter", nh);
 */
tolerance
_loadTolerance(const std::string& _name, ros::NodeHandle& _nh) {
  XmlRpcValue raw;

  // load the data from the param server.
  // if the user does not provide anything, we don't have any tolerance defined
  if (!_nh.getParam(_name, raw))
    return {};

  // the tolerances must be defined as an array
  if (raw.getType() != XmlRpcValue::TypeArray) {
    DP_WARN_LOG(_name << " is not an array");
    return {};
  }

  // allocate space.
  tolerance::list_type impl;
  impl.reserve(raw.size());

  // iterate over the array
  for (int ii = 0; ii != raw.size(); ++ii) {
    // short-cut access to the current element
    const auto& element = raw[ii];

    // get the tag - we promise not to throw anything
    const auto tag = _getElement<std::string>(element, "type", "none");
    const auto pose = _getPose(element);

    // "switch-case" on the different modes
    if (tag == "angle") {
      impl.emplace_back(tolerance::mode::ANGLE, pose);
    }
    else if (tag == "sphere") {
      impl.emplace_back(tolerance::mode::SPHERE, pose);
    }
    else if (tag == "box") {
      impl.emplace_back(tolerance::mode::BOX, pose);
    }
    else {
      DP_INFO_LOG("ignoring the tag \"" << tag << "\"");
    }
  }
  return {impl};
}

gradient_decent::gradient_decent(parameter&& _param) noexcept :
    param_(std::move(_param)) {}

std::pair<float, Eigen::Vector3d>
gradient_decent::solve(const pose_gradient& _pg,
                       const Eigen::Vector3d& _start) const {
  // super simple gradient decent algorithm with a limit on the max step
  // for now we set it to 1 cell size.
  std::pair<float, Eigen::Vector3d> res{0.0F, _start};
  dpose_core::internal::jacobian_data::jacobian J;
  for (size_t ii = 0; ii != param_.iter; ++ii) {
    // get the derivative (d)
    res.first = _pg.get_cost(res.second, &J, nullptr);

    // scale the vector such that its norm is at most the _param.step
    // (the scaling is done seperately for translation (t) and rotation (r))
    const auto norm_t = std::max(J.segment(0, 2).norm(), param_.step_t);
    const auto norm_r = std::max(std::abs(J(2)), param_.step_r);
    J.segment(0, 2) *= (param_.step_t / norm_t);
    J(2) *= (param_.step_r / norm_r);

    // the "gradient decent"
    res.second -= J;
    if (res.first <= param_.epsilon || !param_.tol.within(_start, res.second))
      break;
  }
  return res;
}

// some shortcuts for the conversion functions below
using costmap_2d::Costmap2D;
using Eigen::Vector3d;
using geometry_msgs::PoseStamped;

/// @brief converts PoseStamped to Eigen::Vector3d
Vector3d
_to_eigen(const PoseStamped& _msg) noexcept {
  Vector3d out;
  out.x() = _msg.pose.position.x;
  out.y() = _msg.pose.position.y;
  out.z() = tf2::getYaw(_msg.pose.orientation);
  return out;
}

/// @brief converts Eigen::Vector3d to PoseStamped with the given _frame
PoseStamped
_to_msg(const Vector3d& _se2, const std::string& _frame) noexcept {
  PoseStamped out;
  out.header.frame_id = _frame;
  out.pose.position.x = _se2.x();
  out.pose.position.y = _se2.y();
  // detour to tf2...
  tf2::Quaternion q;
  q.setRPY(0, 0, _se2.z());
  out.pose.orientation = tf2::toMsg(q);
  return out;
}

/// @brief converts metric input to cells using the _map's parameters
Vector3d
_to_cells(const Vector3d& _metric, const Costmap2D& _map) noexcept {
  const Vector3d origin(_map.getOriginX(), _map.getOriginY(), 0);
  const double inv_res = 1. / _map.getResolution();
  return (_metric - origin) * inv_res;
}

/// @brief converts cell to metric output using the _map's parameters
Vector3d
_to_metric(const Vector3d& _cell, const Costmap2D& _map) noexcept {
  const Vector3d origin(_map.getOriginX(), _map.getOriginY(), 0);
  const double res = _map.getResolution();
  return origin + _cell * res;
}

}  // namespace internal

using namespace internal;

bool
DposeGoalTolerance::preProcess(Pose& _start, Pose& _goal) {
  // we will only have a valid costmap ptr, if DposeGoalTolerance::initialize
  // was called successfully.
  if (!map_) {
    DP_WARN_LOG("not initialized");
    return false;
  }

  // check if the _goal is in the same frame as the costmap.
  if (_goal.header.frame_id != map_->getGlobalFrameID()) {
    DP_WARN_LOG("unknown frame_id: " << _goal.header.frame_id);
    return false;
  }

  // convert the metric input to cells - that's what we need for the
  // gradient_decent::solve call.
  const auto& map = *map_->getCostmap();
  const Vector3d metric_se2 = _to_eigen(_goal);
  const Vector3d cell_se2 = _to_cells(metric_se2, map);

  // run the optimization on the goal pose
  const std::pair<float, Vector3d> res = opt_.solve(grad_, cell_se2);

  // we only update the pose, if we have "succeeded"
  const auto success = res.first <= epsilon_;
  if (success)
    _goal = _to_msg(_to_metric(res.second, map), _goal.header.frame_id);

  // report the result
  DP_INFO_LOG("finished with cost " << res.first << " and the pose "
                                    << res.second.transpose());

  return success;
}

void
DposeGoalTolerance::initialize(const std::string& _name, Map* _map) {
  // we cannot do anything if the costmap is invalid
  if (!_map) {
    DP_FATAL_LOG("received a nullptr as map");
    return;
  }

  map_ = _map;
  ros::NodeHandle pnh("~" + _name);

  {
    // scope since we have different parameters
    // here we take care of the pose_gradient setup
    dpose_core::pose_gradient::parameter param;

    // get the padding for the footprint.
    // the padding is unsigned so we take here the abs
    param.padding = std::abs(pnh.param("padding", 2));

    // setup the pose-gradient
    grad_ = dpose_core::pose_gradient{*_map->getLayeredCostmap(), param};
  }

  {
    // here we take care of the gradient_decent setup
    gradient_decent::parameter param;
    // get the tolerances
    param.tol = _loadTolerance("tolerance", pnh);

    // get the other parameters. all parameters cannot be negative, so we take
    // their absolute value
    param.iter = std::abs(pnh.param("iter", 10));
    param.step_r = std::abs(pnh.param("step_r", 0.1));
    param.step_t = std::abs(pnh.param("step_t", 1.));
    epsilon_ = param.epsilon = pnh.param("epsilon", 0.5);

    // setup the gradient-decent
    opt_ = gradient_decent{std::move(param)};
  }
}

}  // namespace dpose_goal_tolerance

PLUGINLIB_EXPORT_CLASS(dpose_goal_tolerance::DposeGoalTolerance,
                       gpp_interface::PrePlanningInterface)
