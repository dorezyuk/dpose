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

#include <angles/angles.h>
#include <costmap_2d/cost_values.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <boost/thread/lock_types.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

/// @brief the name of the module
constexpr char _name[] = "[dpose_goal_tolerance]: ";

// custom prints
#define DP_DEBUG(args) ROS_DEBUG_STREAM(_name << args)
#define DP_INFO_LOG(args) ROS_INFO_STREAM(_name << args)
#define DP_WARN_LOG(args) ROS_WARN_STREAM(_name << args)
#define DP_ERROR_LOG(args) ROS_ERROR_STREAM(_name << args)
#define DP_FATAL_LOG(args) ROS_FATAL_STREAM(_name << args)

namespace dpose_goal_tolerance {

using namespace dpose_core;

pose_regularization::pose_regularization(number _weight_lin,
                                         number _weight_rot) :
    weight_lin_(_weight_lin), weight_rot_(_weight_rot) {
  if (!weight_lin_ && !weight_rot_)
    throw std::invalid_argument("both weights cannot be zero");
}

void
pose_regularization::init(const pose &_start, const pose &_goal) noexcept {
  // get the linear and rotational normalization weights.
  auto w_lin = (_start - _goal).segment(0, 2).squaredNorm();
  auto w_rot =
      std::pow(angles::shortest_angular_distance(_start.z(), _goal.z()), 2);

  // avoid division by zero
  w_lin = weight_lin_ / std::max(w_lin, 1e-3);
  w_rot = weight_rot_ / std::max(w_rot, 1e-3);
  goal_ = _goal;
  norm_ = pose(w_lin, w_lin, w_rot);
  // the hessian is a constant eye matrix of the form
  // [[2 * w_lin, 0, 0],
  //  [0, 2 * w_lin, 0],
  //  [0, 0, 2 * w_rot]]
  H_ = 2 * norm_.asDiagonal();
}

number
pose_regularization::get_cost(const pose &_pose) noexcept {
  // get the distance to the goal
  diff_ = _pose - goal_;
  diff_.z() = angles::normalize_angle(diff_.z());

  // the jacobian is
  // 2 * [w_lin * delta_x, w_lin * delta_y, w_rot * delta_theta]^T.
  J_ = 2 * (norm_.array() * diff_.array()).matrix();

  // the cost is
  // w_lin * delta_x^2 + w_lin * delta_y^2 + w_rot * delta_theta^2
  return norm_.dot(diff_.array().pow(2).matrix());
}

problem::problem(costmap &_map, const pose_gradient::parameter &_param) :
    costmap_(&_map),
    pg_(make_footprint(*_map.getLayeredCostmap()), _param),
    compute_hessian_(_param.generate_hessian) {}

void
problem::on_new_x(index _n, const number *_x) {
  // convert ipopt to eigen
  const pose offset(_x[0], _x[1], _x[2]);
  const pose p = offset + pose_;

  // reset the cost
  cost_ = 0;
  // apply the regularization for start and goal (if defined)
  for (auto &reg : regs_)
    cost_ += reg.get_cost(p);

  // query the data
  if (compute_hessian_) {
    // the case where we use the hessian
    cost_ =
        pg_.get_cost(p, lethal_cells_.begin(), lethal_cells_.end(), &J_, &H_);

    // assemble the hessian: flip the sign and add the hessians from the
    // constraints.
    H_ *= -1;
    for (auto &reg : regs_)
      H_ += reg.get_hessian();
  }
  else
    cost_ = pg_.get_cost(p, lethal_cells_.begin(), lethal_cells_.end(), &J_,
                         nullptr);

  // assemble the jacobian: flip the signs and add the jacobians from the
  // constraints.
  J_ *= -1;
  for (auto &reg : regs_)
    J_ += reg.get_jacobian();
}

void
problem::init(const pose &_start, const pose &_goal,
              const DposeGoalToleranceConfig &_param) {
  pose_ = _goal;
  lin_tol_sq_ = std::pow(_param.lin_tolerance, 2);
  rot_tol_sq_ = std::pow(_param.rot_tolerance, 2);

  // get the bounding box
  const auto kernel_box = pg_.get_data().core.get_box();

  // get the max. thats a simplification to deal with rotations
  const auto max_coeff = kernel_box.array().abs().maxCoeff();

  // add the tolerance to get the halfed size of the rectangle
  const auto h = (max_coeff + std::abs(_param.lin_tolerance));

  // get the final search box
  rectangle<int> search_box;
  // clang-format off
  search_box << -h,  h, h, -h, -h,
                -h, -h, h,  h, -h;
  // clang-format on
  search_box = (search_box.cast<double>().colwise() + pose_.segment(0, 2))
                   .array()
                   .round()
                   .matrix()
                   .cast<int>();

  // now get the lethal cells
  lethal_cells_ = lethal_cells_within(*costmap_->getCostmap(), search_box);

  // setup the additional penalties
  regs_.clear();
  if (_param.weight_goal_lin || _param.weight_goal_rot) {
    regs_.emplace_back(_param.weight_goal_lin, _param.weight_goal_rot);
    // we normalize by the tolerance
    pose start = _goal + pose(_param.lin_tolerance, 0, _param.rot_tolerance);
    regs_.back().init(start, _goal);
  }

  if (_param.weight_start_lin || _param.weight_start_rot) {
    regs_.emplace_back(_param.weight_start_lin, _param.weight_start_rot);
    // the error is the distance to the start - hence the _start pose is the
    // "goal".
    pose start = _start;

    // if the distance between _start and _goal is bigger then the tolerance,
    // we can never reach "zero" costs. we hence place a fake start pose at
    // the goal-tolerance distance from _goal.
    pose diff = _start - _goal;
    diff.z() = angles::normalize_angle(diff.z());

    // get the "norms"
    const auto diff_lin_norm = diff.segment(0, 2).norm();
    const auto diff_rot_norm = std::abs(diff.z());

    // linear part (we drop the segment(0, 2) extensions and corrent the theta
    // below).
    if (diff_lin_norm != 0 && diff_lin_norm > _param.lin_tolerance)
      start = _goal + diff * _param.lin_tolerance / diff_lin_norm;

    // angular part (with theta correction)
    if (diff_rot_norm != 0 && diff_rot_norm > _param.rot_tolerance)
      start.z() = _goal.z() + diff.z() * _param.rot_tolerance / diff_rot_norm;
    else
      start.z() = _start.z();

    regs_.back().init(_goal, start);
  }
}

bool
problem::get_nlp_info(index &_n, index &_m, index &_nonzero_jac_g,
                      index &_nonzero_h_lag, IndexStyleEnum &_index_style) {
  _n = 3;              // we optimize in se2
  _m = 2;              // contraints in translation and rotation
  _nonzero_jac_g = 3;  // we just have g0 / dx0, g0 / dy and g1 /dtheta
  _nonzero_h_lag = 6;  // dense hessian with just lower half
  _index_style = C_STYLE;
  return true;
}

bool
problem::get_bounds_info(index _n, number *_x_lower, number *_x_upper, index _m,
                         number *_g_lower, number *_g_upper) {
  for (index nn = 0; nn != _n; ++nn) {
    _x_lower[nn] = -1e6;
    _x_upper[nn] = 1e6;
  }
  _g_lower[0] = 0;
  _g_lower[1] = 0;
  _g_upper[0] = lin_tol_sq_;
  _g_upper[1] = rot_tol_sq_;
  return true;
}

bool
problem::get_starting_point(index _n, bool _init_x, number *_x, bool _init_z,
                            number *_z_L, number *_z_U, index _m,
                            bool _init_lambda, number *lambda) {
  std::fill_n(_x, _n, 0);
  return true;
}

bool
problem::eval_f(index _n, const number *_x, bool _new_x, number &_f_value) {
  if (_new_x)
    on_new_x(_n, _x);
  _f_value = cost_;
  return true;
}

bool
problem::eval_grad_f(index _n, const number *_x, bool _new_x, number *_grad_f) {
  if (_new_x)
    on_new_x(_n, _x);
  std::copy_n(J_.data(), _n, _grad_f);
  return true;
}

bool
problem::eval_g(index _n, const number *_x, bool _new_x, index _m, number *_g) {
  if (_new_x)
    on_new_x(_n, _x);
  // get the squared diff
  const pose diff = pose(_x[0], _x[1], _x[2]).array().pow(2).matrix();

  // write into the output
  _g[0] = diff(0) + diff(1);
  _g[1] = diff(2);
  return true;
}

bool
problem::eval_jac_g(index _n, const number *_x, bool _new_x, index _m,
                    index _number_elem_jac, index *_i_row, index *_j_col,
                    number *_jac_g) {
  if (_new_x)
    on_new_x(_n, _x);

  if (!_jac_g) {
    index ii = 0;
    /* the jacobian looks like
     * [[g0 / dx, g0 / dy, 0],
     *  [0,     , 0,     , g1 / dtheta]]
     */
    _i_row[ii] = 0;
    _j_col[ii] = 0;

    _i_row[++ii] = 0;
    _j_col[ii] = 1;

    _i_row[++ii] = 1;
    _j_col[ii] = 2;
    assert(++ii == _number_elem_jac);
  }
  else {
    // recap: our constraints are defined as
    // g0 = x^2 + y^2
    // g1 = theta^2
    //
    // the derivatives are then
    // 0: g0 / dx = 2 * x
    // 1: g0 / dy = 2 * y
    // 2: g1 / dtheta = 2 * theta
    const pose diff = pose(_x[0], _x[1], _x[2]) * 2;
    std::copy_n(diff.data(), _number_elem_jac, _jac_g);
  }

  return true;
}

bool
problem::eval_h(index _n, const number *_x, bool _new_x, number _obj_factor,
                index _m, const number *_lambda, bool _new_lambda,
                index _number_elem_hess, index *_i_row, index *_j_col,
                number *_hess) {
  if (_new_x)
    on_new_x(_n, _x);

  index idx = 0;
  if (!_hess) {
    // define the ordering for the dense hessian. the order goes like
    // 0
    // 1 2
    // 3 4 5
    for (index row = 0; row != _n; ++row) {
      for (index col = 0; col <= row; ++col, ++idx) {
        _i_row[idx] = row;
        _j_col[idx] = col;
      }
    }
  }
  else {
    // g0 / (dx dx) = 2
    // g0 / (dy dy) = 2
    // g1 / (dtheta dtheta) = 2
    // all other elements are zero
    const pose diag(_lambda[0], _lambda[0], _lambda[1]);
    const pose_gradient::hessian H_g = 2 * diag.asDiagonal();
    const pose_gradient::hessian H_lag = _obj_factor * H_ + H_g;

    // write the hessian into the output buffer
    _hess[idx++] = H_lag(0, 0);  // lag / (dx dx)
    _hess[idx++] = H_lag(1, 0);  // lag / (dy dx)
    _hess[idx++] = H_lag(1, 1);  // lag / (dy dy)
    _hess[idx++] = H_lag(2, 0);  // lag / (dtheta dx)
    _hess[idx++] = H_lag(2, 1);  // lag / (dtheta dy)
    _hess[idx++] = H_lag(2, 2);  // lag / (dtheta dtheta)
  }
  assert(idx == _number_elem_hess);
  return true;
}

void
problem::finalize_solution(Ipopt::SolverReturn _status, index _n,
                           const number *_x, const number *_z_L,
                           const number *_z_U, index _m, const number *_g,
                           const number *_lambda, number _obj_value,
                           const Ipopt::IpoptData *_ip_data,
                           Ipopt::IpoptCalculatedQuantities *_ip_cq) {
  const pose p(_x[0], _x[1], _x[2]);
  pose_ += p;
}

pose
to_cell(const DposeGoalTolerance::Pose &_pose,
        const DposeGoalTolerance::Map &_map) {
  // check if the _goal is in the same frame as the costmap.
  if (_pose.header.frame_id != _map.getGlobalFrameID())
    throw std::runtime_error("pose not in the global map frame");

  const auto &p = _pose.pose.position;
  unsigned int x, y;
  if (!_map.getCostmap()->worldToMap(p.x, p.y, x, y))
    throw std::runtime_error("pose outside of the map");

  // get yaw
  const auto yaw = tf2::getYaw(_pose.pose.orientation);
  return pose{static_cast<number>(x), static_cast<number>(y), yaw};
}

DposeGoalTolerance::Pose
to_msg(const pose &_pose, const DposeGoalTolerance::Map &_map) {
  // check to prevent underflow
  if ((_pose.segment(0, 2).array() < 0).any())
    throw std::runtime_error("negative pose");

  // convert to unsigned int
  unsigned int x = std::round(_pose.x());
  unsigned int y = std::round(_pose.y());

  DposeGoalTolerance::Pose msg;

  // get the metric data
  _map.getCostmap()->mapToWorld(x, y, msg.pose.position.x, msg.pose.position.y);
  tf2::Quaternion q;
  q.setRPY(0, 0, _pose.z());
  msg.pose.orientation = tf2::toMsg(q);

  return msg;
}

bool
DposeGoalTolerance::preProcess(Pose &_start, Pose &_goal) {
  // this class is a noop if both tolerances are zero
  if (param_.lin_tolerance <= 0 && param_.rot_tolerance <= 0) {
    DP_INFO_LOG("class disabled");
    return true;
  }

  // we will only have a valid costmap ptr, if DposeGoalTolerance::initialize
  // was called successfully.
  if (!map_) {
    DP_WARN_LOG("not initialized");
    return false;
  }

  // convert the metric input to cells - that's what we need for the solver
  pose c_goal, c_start;
  try {
    c_goal = to_cell(_goal, *map_);
    c_start = to_cell(_start, *map_);
  }
  catch (std::runtime_error &_ex) {
    DP_WARN_LOG("cannot transform to cells: " << _ex.what());
    return false;
  }

  auto our_problem = dynamic_cast<problem *>(Ipopt::GetRawPtr(problem_));
  our_problem->init(c_start, c_goal, param_);

  auto status = solver_->OptimizeTNLP(problem_);

  try {
    _goal.pose = to_msg(our_problem->get_pose(), *map_).pose;
  }
  catch (std::runtime_error &_ex) {
    DP_WARN_LOG("cannot convert to message: " << _ex.what());
    return false;
  }

  // publish the filtered pose
  if (status == Ipopt::Solve_Succeeded) {
    pose_pub_.publish(_goal);
    return false;
  }
  return false;
}

using solver_ptr = Ipopt::SmartPtr<Ipopt::IpoptApplication>;

inline void
load_ipopt_cfg(solver_ptr &_solver, ros::NodeHandle &_nh,
               const std::string &_name, int _default) {
  _solver->Options()->SetIntegerValue(_name, _nh.param(_name, _default));
}

inline void
load_ipopt_cfg(solver_ptr &_solver, ros::NodeHandle &_nh,
               const std::string &_name, double _default) {
  _solver->Options()->SetNumericValue(_name, _nh.param(_name, _default));
}

inline void
load_ipopt_cfg(solver_ptr &_solver, ros::NodeHandle &_nh,
               const std::string &_name, const std::string &_default) {
  _solver->Options()->SetStringValue(_name, _nh.param(_name, _default));
}

void
DposeGoalTolerance::initialize(const std::string &_name, Map *_map) {
  // we cannot do anything if the costmap is invalid
  if (!_map) {
    DP_FATAL_LOG("received a nullptr as map");
    return;
  }

  map_ = _map;
  ros::NodeHandle nh("~" + _name);

  // read the padding parameter and safely cast it to unsigned int
  const int padding = nh.param("padding", 2);
  pose_gradient::parameter param;
  param.padding = static_cast<unsigned int>(std::max(padding, 0));
  // the hessian from dpose is very noisy for most footprints and its better
  // not to use it.
  param.generate_hessian = nh.param("use_hessian", false);
  if (param.generate_hessian)
    DP_WARN_LOG("Hessian from dpose-core is very noisy");

  problem_ = new problem(*map_, param);
  solver_ = IpoptApplicationFactory();

  // configure the solver
  load_ipopt_cfg(solver_, nh, "tol", 5.);
  load_ipopt_cfg(solver_, nh, "mu_strategy", "adaptive");
  load_ipopt_cfg(solver_, nh, "output_file", "/tmp/ipopt.out");
  load_ipopt_cfg(solver_, nh, "max_iter", 20);
  load_ipopt_cfg(solver_, nh, "max_cpu_time", .5);

  // tell ipopt to use quasi-newtow method if we don't use the Hessian from
  // dpose.
  if (!param.generate_hessian)
    solver_->Options()->SetStringValue("hessian_approximation",
                                       "limited-memory");
// print the derivative test if running in debug
#ifndef NDEBUG
  solver_->Options()->SetStringValue("derivative_test", "first-order");
  solver_->Options()->SetNumericValue("derivative_test_perturbation", 0.00001);
  solver_->Options()->SetNumericValue("point_perturbation_radius", 10);
#endif

  pose_pub_ = nh.advertise<Pose>("filtered", 1);

  // setup the cfg-server
  cfg_server_.reset(new cfg_server(nh));
  cfg_server_->setCallback(
      [&](DposeGoalToleranceConfig &_cfg, uint32_t _level) {
        param_ = _cfg;
        // lin is in cells
        param_.lin_tolerance /= map_->getCostmap()->getResolution();
      });
}

}  // namespace dpose_goal_tolerance

PLUGINLIB_EXPORT_CLASS(dpose_goal_tolerance::DposeGoalTolerance,
                       gpp_interface::PrePlanningInterface)
