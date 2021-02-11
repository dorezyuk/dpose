#include <dpose_core/dpose_costmap.hpp>
#include <dpose_recovery/dpose_recovery.hpp>

#include <pluginlib/class_list_macros.hpp>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Twist.h>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <cassert>
#include <iterator>
#include <limits>
#include <string>

namespace dpose_recovery {

inline size_t
little_gauss(size_t _n) noexcept {
  return _n * (_n + 1) / 2;
}

Problem::Problem(costmap_2d::Costmap2DROS &_cm, const Parameter &_our_param,
                 const pose_gradient::parameter &_param) :
    pg_(dpose_core::make_footprint(*_cm.getLayeredCostmap()), _param),
    param_(_our_param),
    x_(_our_param.steps),
    R(_our_param.steps),
    T(_our_param.steps),
    R_hat(_our_param.steps),
    J_hat(_our_param.steps),
    J_tilde(_our_param.steps),
    H_hat(_our_param.steps),
    C_hat(_our_param.steps),
    D_hat(_our_param.steps),
    u_best(_our_param.steps),
    map_(&_cm) {}

bool
Problem::get_nlp_info(index &n, index &m, index &nnz_jac_g, index &nnz_h_lag,
                      IndexStyleEnum &index_style) {
  // number of variables
  // dim_u should depend on the robot model (diff-drive vs omni-drive)
  n = param_.steps * param_.dim_u;

  // for now: no constraints. later on we might add maximum distance for the
  // recovery.
  m = nnz_jac_g = 0;

  // our hessian is full. since we populate only the lower half of a NxN matrix,
  // we can use the little-gauss formula to find the number of entries.
  nnz_h_lag = little_gauss(n);

  index_style = TNLP::C_STYLE;
  return true;
};

bool
Problem::get_bounds_info(index n, number *u_l, number *u_u, index m,
                         number *g_l, number *g_u) {
  size_t ll = 0;
  size_t uu = 0;
  for (size_t ii = 0; ii != param_.steps; ++ii) {
    u_u[uu++] = param_.u_upper.x();
    u_u[uu++] = param_.u_upper.y();
    u_l[ll++] = param_.u_lower.x();
    u_l[ll++] = param_.u_lower.y();
  }

  assert(ll == n && "bad ll index");
  assert(uu == n && "bad uu index");

  return true;
};

bool
Problem::get_starting_point(index n, bool init_x, number *x, bool init_z,
                            number *z_L, number *z_U, index m, bool init_lambda,
                            number *lambda) {
  for (index nn = 0; nn != n; ++nn)
    x[nn] = 0;

  cost_best = std::numeric_limits<number>::max();
  return true;
};

template <typename _Iter>
void
_create_hat(_Iter _begin, _Iter _end) {
  if (_begin == _end)
    return;

  for (_Iter next = std::next(_begin); next != _end; ++next, ++_begin)
    *next += *_begin;
}

using namespace dpose_core;

cell_rectangle
get_bounding_box(index n, const number *u, pose _prev,
                 const cell_rectangle &_box) {
  if (!n)
    throw std::invalid_argument("n cannot be zero");

  pose curr;
  // create the vectors for the min and max corners
  Eigen::Vector2d max_corner(std::numeric_limits<double>::lowest(),
                             std::numeric_limits<double>::lowest());
  Eigen::Vector2d min_corner(std::numeric_limits<double>::max(),
                             std::numeric_limits<double>::max());
  // iterate over the vector of controls
  for (index ii = 0; ii != n; ii += 2) {
    // get the current pose
    curr = diff_drive::step(_prev, u[ii], u[ii + 1]);

    // transform the box into the pose
    const Eigen::Translation2d trans(curr.x(), curr.y());
    const Eigen::Rotation2Dd rot(curr.z());
    const Eigen::Isometry2d tf = trans * rot;
    const rectangle<double> curr_box = tf * _box.cast<double>();

    // update the corners
    Eigen::Vector2d max = curr_box.rowwise().maxCoeff();
    Eigen::Vector2d min = curr_box.rowwise().minCoeff();
    max_corner = max_corner.array().max(max.array());
    min_corner = min_corner.array().min(min.array());

    _prev = curr;
  }

  return to_rectangle(min_corner, max_corner).cast<int>();
}

void
Problem::on_new_u(index n, const number *u) {
  using state_t = Eigen::Vector3d;
  // we will throw otherwise
  if (!n)
    return;

  // define the states
  state_t prev = x0_;
  state_t curr;

  // get the bounding gox of the entire motion
  const rectangle<int> box =
      get_bounding_box(n, u, x0_, pg_.get_data().core.get_box());

  // get the lethal cells
  const auto lethal_cells = lethal_cells_within(*map_->getCostmap(), box);
  // reset the cost
  cost = 0;

  // jj is the index within u, which has the form [v0, w0, v1, w1,..., wN]
  for (size_t ii = 0, jj = 0; ii != param_.steps; ++ii, jj += 2) {
    // now get the jacobians of the current state T_ii and R_ii
    // note: at this step we safe R_ii into the R_hat_ii buffer
    diff_drive::T_jacobian(prev, T.at(ii));
    diff_drive::R_jacobian(prev.z(), u[jj + 1], R_hat.at(ii));

    // get the state x_n = A x_{n-1} + B u_n
    curr = diff_drive::step(prev, u[jj], u[jj + 1]);

    // find J_ii and H_ii. We abuse the J_hat and H_hat as dummies.
    cost += pg_.get_cost(curr, lethal_cells.begin(), lethal_cells.end(),
                         &J_hat.at(ii), &H_hat.at(ii));

    J_hat.at(ii) *= -1;
    H_hat.at(ii) *= -1;
    prev = curr;
  }
  if (cost < cost_best) {
    // save x
    for (size_t uu = 0, nn = 0; uu != param_.steps; ++uu, nn += 2)
      u_best.at(uu) << u[nn], u[nn + 1];
    cost_best = cost;
  }

  ROS_WARN_STREAM("cost " << cost);

  // init the R_hat
  _create_hat(R_hat.begin(), R_hat.end());

  // init J_tilde, C_hat and D_hat todo rename to P_hat
  for (size_t ii = 0; ii != param_.steps; ++ii)
    J_tilde.at(ii) = R_hat.at(ii).transpose() * J_hat.at(ii);

  for (size_t ii = 0; ii != param_.steps; ++ii)
    C_hat.at(ii) = H_hat.at(ii) * R_hat.at(ii);

  for (size_t ii = 0; ii != param_.steps; ++ii)
    D_hat.at(ii) = R_hat.at(ii).transpose() * C_hat.at(ii);

  _create_hat(C_hat.begin(), C_hat.end());
  _create_hat(D_hat.begin(), D_hat.end());
  _create_hat(J_tilde.begin(), J_tilde.end());

  // init J_hat and H_hat
  _create_hat(H_hat.begin(), H_hat.end());
  _create_hat(J_hat.begin(), J_hat.end());
}

bool
Problem::eval_f(index n, const number *u, bool new_u, number &_cost) {
  if (new_u)
    on_new_u(n, u);
  _cost = cost;
  return true;
}

bool
Problem::eval_grad_f(index n, const number *u, bool new_u, number *grad_f) {
  if (new_u)
    on_new_u(n, u);

  using gradient = Eigen::Matrix<number, diff_drive::u_dim, 1UL>;
  diff_drive::jacobian T_tilde;
  gradient grad;

  // catch the degenerate case - we need front() and back() to be defined on our
  // vectors.
  if (T.empty())
    return false;

  // the outout grad_f hat the form of
  // [df / dv_0, df / dw_O, df / dv_1, df /dw_1, ... df /dw_N]
  // we will use dd as the index on the grad_f (dd as in destination)
  size_t dd = 0;

  // the first step is somewhat special since the minus terms are missing.
  T_tilde = T.front() - R_hat.front();

  // grad contains the gradient of f with respect to v_0 and w_0
  grad = T_tilde.transpose() * J_hat.back() + J_tilde.back();
  grad_f[dd] = grad(0);
  grad_f[++dd] = grad(1);

  for (size_t ii = 1; ii != param_.steps; ++ii) {
    T_tilde = T.at(ii) - R_hat.at(ii);
    // grad contains the gradient of f with respect to v_ii and w_ii
    grad = T_tilde.transpose() * (J_hat.back() - J_hat.at(ii - 1)) +
           (J_tilde.back() - J_tilde.at(ii - 1));
    grad_f[++dd] = grad(0);
    grad_f[++dd] = grad(1);
  }

  // make sure that we have reached the end
  assert(++dd == n && "failed to populate the gradient");

  return true;
};

bool
Problem::eval_g(index n, const number *x, bool new_x, index m, number *g) {
  return true;
};

bool
Problem::eval_jac_g(index n, const number *x, bool new_x, index m,
                    index nele_jac, index *iRow, index *jCol, number *values) {
  return true;
};

bool
Problem::eval_h(index n, const number *u, bool new_u, number obj_factor,
                index m, const number *lambda, bool new_lambda, index nele_hess,
                index *iRow, index *jCol, number *values) {
  ROS_INFO_STREAM("CALLING " << __FUNCTION__);

  if (new_u)
    on_new_u(n, u);

  if (!values) {
    // init the rows and cols
    // [[df / (du0 du0), df / (du0 dw0), ..., df / (du0 dwN)],
    //  [df / (dw0 du0), df / (dw0 dw0), ..., df / (dw0 dwN)],
    //  [df / (du1 du0), df / (du1 dw0), ..., df / (du1 dwN)],
    //  ...
    //  [df / (dwN du0), df / (dwN dw0), ..., df / (dwN dwN)]]
    index idx = 0;
    for (index row = 0; row != n; ++row)
      for (index col = 0; col <= row; ++col, ++idx) {
        iRow[idx] = row;
        jCol[idx] = col;
      }

    assert(idx == nele_hess && "wrong index");
  }
  else {
    // populate the hessian
    size_t uu = 0;
    size_t ww = 0;
    diff_drive::jacobian T_tilde_rr, T_tilde_cc, C_hat_diff;

    // the hess matrix is [[df / (dv_r dv_c), df / (dv_r dw_c)],
    //                     [df / (dw_r dv_c), df / (dw_r dw_c)]]
    Eigen::Matrix<number, 2UL, 2UL> hess;

    // first element is "special" since there is no o-1 index
    T_tilde_rr = T.front() - R_hat.front();
    C_hat_diff = C_hat.back();
    // clang-format off
    hess = T_tilde_rr.transpose() * H_hat.back() * T_tilde_rr +
           T_tilde_rr.transpose() * C_hat_diff +
           C_hat_diff.transpose() * T_tilde_rr +
           D_hat.back();
    // clang-format on

    values[uu] = hess(0, 0);
    values[++ww] = hess(1, 0);  // ww is 1
    values[++ww] = hess(1, 1);  // ww is 2

    for (size_t rr = 1; rr != param_.steps; ++rr) {
      uu = ww;
      ww += (rr * 2) + 1;
      for (size_t cc = 0; cc != rr; ++cc) {
        T_tilde_rr = T.at(rr) - R_hat.at(rr);
        T_tilde_cc = T.at(cc) - R_hat.at(cc);
        C_hat_diff = C_hat.back() - C_hat.at(rr - 1);
        // clang-format off
        hess = T_tilde_rr.transpose() * (H_hat.back() - H_hat.at(rr - 1)) * T_tilde_cc +
               T_tilde_rr.transpose() * C_hat_diff +
               C_hat_diff.transpose() * T_tilde_cc +
               (D_hat.back() - D_hat.at(rr - 1));
        // clang-format on

        // copy the hess-values into the destination
        values[++uu] = hess(0, 0);
        values[++uu] = hess(0, 1);
        values[++ww] = hess(1, 0);
        values[++ww] = hess(1, 1);
      }
      // last element is "special" since we just need three out of four values
      T_tilde_rr = T.at(rr) - R_hat.at(rr);
      C_hat_diff = C_hat.back() - C_hat.at(rr - 1);
      // clang-format off
      hess = T_tilde_rr.transpose() * (H_hat.back() - H_hat.at(rr - 1)) * T_tilde_rr +
              T_tilde_rr.transpose() * C_hat_diff +
              C_hat_diff.transpose() * T_tilde_rr +
              (D_hat.back() - D_hat.at(rr - 1));
      // clang-format on

      values[++uu] = hess(0, 0);
      values[++ww] = hess(1, 0);
      values[++ww] = hess(1, 1);
    }
  }

  return true;
}

void
Problem::finalize_solution(Ipopt::SolverReturn status, index n, const number *x,
                           const number *z_L, const number *z_U, index m,
                           const number *g, const number *lambda,
                           number obj_value, const Ipopt::IpoptData *ip_data,
                           Ipopt::IpoptCalculatedQuantities *ip_cq) {
  // save x
  for (size_t uu = 0, nn = 0; uu != param_.steps; ++uu, nn += 2)
    u_best.at(uu) << x[nn], x[nn + 1];
};

using solver_ptr = Ipopt::SmartPtr<Ipopt::IpoptApplication>;

inline void
load_ipopt(solver_ptr &_solver, ros::NodeHandle &_nh, const std::string &_name,
           int _default) {
  _solver->Options()->SetIntegerValue(_name, _nh.param(_name, _default));
}

inline void
load_ipopt(solver_ptr &_solver, ros::NodeHandle &_nh, const std::string &_name,
           double _default) {
  _solver->Options()->SetNumericValue(_name, _nh.param(_name, _default));
}

inline void
load_ipopt(solver_ptr &_solver, ros::NodeHandle &_nh, const std::string &_name,
           const std::string &_default) {
  _solver->Options()->SetStringValue(_name, _nh.param(_name, _default));
}

void
DposeRecovery::initialize(std::string _name, tf2_ros::Buffer *_tf,
                          Map *_global_map, Map *_local_map) {
  tf_ = _tf;
  map_ = _local_map;
  Problem::Parameter param;
  ros::NodeHandle nh("~" + _name);

  param.steps = nh.param("steps", 10);
  param.dim_u = 2;

  const auto lin_vel = nh.param("lin_vel", 0.1) / map_->getCostmap()->getResolution();
  const auto rot_vel = nh.param("rot_vel", 0.1);
  param.u_lower = {-lin_vel, -rot_vel};
  param.u_upper = {lin_vel, rot_vel};
  dpose_core::pose_gradient::parameter param2;
  param2.generate_hessian = true;
  param2.padding = nh.param("padding", 5);

  problem_ = new Problem(*map_, param, param2);
  solver_ = IpoptApplicationFactory();

  load_ipopt(solver_, nh, "tol", 5.);
  load_ipopt(solver_, nh, "mu_strategy", "adaptive");
  load_ipopt(solver_, nh, "output_file", "/tmp/ipopt.out");
  load_ipopt(solver_, nh, "max_iter", 5);
  load_ipopt(solver_, nh, "max_cpu_time", .2);

  if (nh.param("hessian_approximation", false))
    solver_->Options()->SetStringValue("hessian_approximation",
                                       "limited-memory");

  // Initialize the IpoptApplication and process the options
  auto status = solver_->Initialize();
  if (status != Ipopt::Solve_Succeeded)
    throw std::runtime_error("ipopt-initialization failed");

  pose_array_ = nh.advertise<geometry_msgs::PoseArray>("poses", 1);
  cmd_vel_ = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 1);
}

void
DposeRecovery::runBehavior() {
  auto problem = dynamic_cast<Problem *>(problem_.operator->());
  geometry_msgs::PoseStamped robot_pose;

  const auto cm = map_->getLayeredCostmap()->getCostmap();
  const auto origin_x = cm->getOriginX();
  const auto origin_y = cm->getOriginY();
  const auto res = cm->getResolution();

  const auto end_time = ros::Time::now() + ros::Duration(10);
  ros::Rate spin_rate(10);

  solver_->Options()->SetStringValue("warm_start_init_point", "no");

  while (ros::Time::now() < end_time) {
    if (!map_->getRobotPose(robot_pose))
      ROS_WARN("Failed to get the robot pose");

    dpose_core::pose_gradient::pose pose;
    pose.x() = (robot_pose.pose.position.x - origin_x) / res;
    pose.y() = (robot_pose.pose.position.y - origin_y) / res;
    pose.z() = tf2::getYaw(robot_pose.pose.orientation);

    problem->set_origin(pose);

    auto status = solver_->OptimizeTNLP(problem_);

    const auto &u = problem->get_u();

    // for(const auto& uu : u)
    //   std::cout << uu.transpose() << std::endl;

    std::vector<Eigen::Vector3d> x(u.size() + 1);

    x.front() = pose;
    for (size_t ii = 0; ii != u.size(); ++ii){
      x.at(ii + 1) = diff_drive::step(x.at(ii), u.at(ii)(0), u.at(ii)(1));
      // std::cout << x.at(ii + 1).transpose() << std::endl;
    }

    // convert to array
    geometry_msgs::PoseArray pose_array;
    tf2::Quaternion q;
    pose_array.poses.resize(x.size());
    pose_array.header.frame_id = robot_pose.header.frame_id;

    for (size_t ii = 0; ii != x.size(); ++ii) {
      pose_array.poses.at(ii).position.x = x.at(ii)(0) * res + origin_x;
      pose_array.poses.at(ii).position.y = x.at(ii)(1) * res + origin_y;
      q.setRPY(0, 0, x.at(ii).z());
      pose_array.poses.at(ii).orientation = tf2::toMsg(q);
      // std::cout << pose_array.poses.at(ii) << std::endl;
    }

    pose_array_.publish(pose_array);

    // convet to twist
    geometry_msgs::Twist cmd_vel;
    cmd_vel.linear.x = u.front().x() * res;
    cmd_vel.angular.z = u.front().y();
    for(size_t ii = 0; ii != 10; ++ii){
      cmd_vel_.publish(cmd_vel);
      spin_rate.sleep();
    }

    solver_->Options()->SetStringValue("warm_start_init_point", "yes");
    // if (problem->get_cost() < 1)
    //   break;
  }
}

}  // namespace dpose_recovery

PLUGINLIB_EXPORT_CLASS(dpose_recovery::DposeRecovery,
                       nav_core::RecoveryBehavior);
