#include <dpose_recovery/dpose_recovery.hpp>

#include <pluginlib/class_list_macros.hpp>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <cassert>
#include <iterator>
#include <string>

namespace dpose_recovery {
namespace internal {

inline size_t
little_gauss(size_t _n) noexcept {
  return _n * (_n + 1) / 2;
}

Problem::Problem(costmap_2d::LayeredCostmap &_lcm, const Parameter &_our_param,
                 const pose_gradient::parameter &_param) :
    pg_(_lcm, _param),
    param_(_our_param),
    x_(_our_param.steps),
    pg_data(_our_param.steps),
    R(_our_param.steps),
    T(_our_param.steps),
    R_hat(_our_param.steps),
    J_hat(_our_param.steps),
    J_tilde(_our_param.steps),
    H_hat(_our_param.steps),
    C_hat(_our_param.steps),
    D_hat(_our_param.steps),
    u(_our_param.steps) {}

bool
Problem::get_nlp_info(Index &n, Index &m, Index &nnz_jac_g, Index &nnz_h_lag,
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
Problem::get_bounds_info(Index n, Number *u_l, Number *u_u, Index m,
                         Number *g_l, Number *g_u) {
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
Problem::get_starting_point(Index n, bool init_x, Number *x, bool init_z,
                            Number *z_L, Number *z_U, Index m, bool init_lambda,
                            Number *lambda) {
  for (Index nn = 0; nn != n; ++nn)
    x[nn] = 0;

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

void
Problem::on_new_u(Index n, const Number *u) {
  using state_t = Eigen::Vector3d;

  for (size_t nn = 0; nn != n; ++nn) {
    std::cout << "(" << u[nn] << ", " << u[nn + 1] << ")" << std::endl;
    ++nn;
  }

  // define the states
  state_t prev = x0_;
  state_t curr;

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
    ROS_INFO_STREAM("curr " << curr.transpose());

    // find J_ii and H_ii. We abuse the J_hat and H_hat as dummies.
    cost += pg_.get_cost(curr, &J_hat.at(ii), &H_hat.at(ii));
    // ROS_INFO_STREAM("J\n" << J_hat.at(ii).transpose());
    // ROS_INFO_STREAM("H\n" << H_hat.at(ii));

    prev = curr;
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
Problem::eval_f(Index n, const Number *u, bool new_u, Number &_cost) {
  if (new_u)
    on_new_u(n, u);
  _cost = cost;
  return true;
}

bool
Problem::eval_grad_f(Index n, const Number *u, bool new_u, Number *grad_f) {
  if (new_u)
    on_new_u(n, u);

  using gradient = Eigen::Matrix<Number, diff_drive::u_dim, 1UL>;
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

  std::cout << "grad ii " << 0 << ": " << grad.transpose() << std::endl;

  for (size_t ii = 1; ii != param_.steps; ++ii) {
    T_tilde = T.at(ii) - R_hat.at(ii);
    // grad contains the gradient of f with respect to v_ii and w_ii
    grad = T_tilde.transpose() * (J_hat.back() - J_hat.at(ii - 1)) +
           (J_tilde.back() - J_tilde.at(ii - 1));
    grad_f[++dd] = grad(0);
    grad_f[++dd] = grad(1);

    std::cout << "grad ii " << ii << ": " << grad.transpose() << std::endl;
  }

  // make sure that we have reached the end
  assert(++dd == n && "failed to populate the gradient");

  for (size_t ii = 0; ii != n; ++ii)
    std::cout << grad_f[ii] << ", ";
  std::cout << std::endl;
  return true;
};

bool
Problem::eval_g(Index n, const Number *x, bool new_x, Index m, Number *g) {
  return true;
};

bool
Problem::eval_jac_g(Index n, const Number *x, bool new_x, Index m,
                    Index nele_jac, Index *iRow, Index *jCol, Number *values) {
  return true;
};

bool
Problem::eval_h(Index n, const Number *u, bool new_u, Number obj_factor,
                Index m, const Number *lambda, bool new_lambda, Index nele_hess,
                Index *iRow, Index *jCol, Number *values) {
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
    Index idx = 0;
    for (Index row = 0; row != n; ++row)
      for (Index col = 0; col <= row; ++col, ++idx) {
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
    Eigen::Matrix<Number, 2UL, 2UL> hess;

    // first element is "special" since there is no o-1 index
    T_tilde_rr = T.front() - R_hat.front();
    C_hat_diff = C_hat.back();
    // clang-format off
    hess = T_tilde_rr.transpose() * H_hat.back() * T_tilde_rr +
           T_tilde_rr.transpose() * C_hat_diff +
           C_hat_diff.transpose() * T_tilde_rr +
           D_hat.back();
    // clang-format on
    std::cout << "J_hat.back()\n" << J_hat.back() << std::endl;
    std::cout << "H_hat.back()\n" << H_hat.back() << std::endl;
    std::cout << "T_tilde_rr\n" << T_tilde_rr << std::endl;
    std::cout << "C_hat_diff\n" << C_hat_diff << std::endl;
    std::cout << "D_hat.back()\n" << D_hat.back() << std::endl;

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
    const auto n_size = little_gauss(n);
    size_t nn = 0;
    for (size_t rr = 0; rr != n; ++rr) {
      for (size_t cc = 0; cc <= rr; ++cc, ++nn) {
        std::cout << values[nn] << ", ";
      }
      std::cout << std::endl;
    }
  }

  return true;
}

void
Problem::finalize_solution(SolverReturn status, Index n, const Number *x,
                           const Number *z_L, const Number *z_U, Index m,
                           const Number *g, const Number *lambda,
                           Number obj_value, const IpoptData *ip_data,
                           IpoptCalculatedQuantities *ip_cq) {
  // save x
  for (size_t uu = 0, nn = 0; uu != param_.steps; ++uu, nn += 2)
    u.at(uu) << x[nn], x[nn + 1];
};
}  // namespace internal

void
DposeRecovery::initialize(std::string _name, tf2_ros::Buffer *_tf,
                          Map *_global_map, Map *_local_map) {
  tf_ = _tf;
  map_ = _local_map;
  internal::Problem::Parameter param;
  param.steps = 5;
  param.dim_u = 2;
  param.u_lower = {-2, -0.1};
  param.u_upper = {2, 0.1};
  dpose_core::pose_gradient::parameter param2;
  param2.generate_hessian = true;
  param2.padding = 5;

  problem_ = new internal::Problem(*map_->getLayeredCostmap(), param, param2);
  solver_ = IpoptApplicationFactory();

  solver_->Options()->SetNumericValue("tol", 1e-7);
  solver_->Options()->SetStringValue("mu_strategy", "adaptive");
  solver_->Options()->SetStringValue("output_file", "/tmp/ipopt.out");
  // solver_->Options()->SetStringValue("derivative_test", "first-order");
  // solver_->Options()->SetIntegerValue("print_level", 10);
  solver_->Options()->SetIntegerValue("max_iter", 2);
  solver_->Options()->SetIntegerValue("max_cpu_time", 1);

  solver_->Options()->SetStringValue("print_user_options", "yes");

  // solver_->Options()->SetStringValue("option_file_name", "/tmp/ipopt.opt");

  // Initialize the IpoptApplication and process the options
  auto status = solver_->Initialize();
  if (status != Ipopt::Solve_Succeeded) {
    std::cout << std::endl
              << std::endl
              << "*** Error during initialization!" << std::endl;
    return;
  }
}

void
DposeRecovery::runBehavior() {
  auto problem = dynamic_cast<internal::Problem *>(problem_.operator->());
  geometry_msgs::PoseStamped robot_pose;
  if (!map_->getRobotPose(robot_pose))
    ROS_WARN("Failed to get the robot pose");

  ROS_INFO_STREAM("robot pose " << robot_pose);

  auto cm = map_->getLayeredCostmap()->getCostmap();
  const auto origin_x = cm->getOriginX();
  const auto origin_y = cm->getOriginY();
  const auto res = cm->getResolution();

  ROS_INFO_STREAM("origin_x " << origin_x);
  ROS_INFO_STREAM("origin_y " << origin_y);

  dpose_core::pose_gradient::pose pose;
  pose.x() = (robot_pose.pose.position.x - origin_x) / res;
  pose.y() = (robot_pose.pose.position.y - origin_y) / res;
  pose.z() = tf2::getYaw(robot_pose.pose.orientation);

  ROS_INFO_STREAM("ORIGIN SET TO " << pose.transpose());
  problem->set_origin(pose);

  ROS_INFO_STREAM("Solving the problem");
  auto status = solver_->OptimizeTNLP(problem_);

  if (status == Ipopt::Solve_Succeeded) {
    std::cout << std::endl
              << std::endl
              << "*** The problem solved!" << std::endl;
  }
  else {
    std::cout << std::endl
              << std::endl
              << "*** The problem FAILED!" << std::endl;
  }
}

}  // namespace dpose_recovery

PLUGINLIB_EXPORT_CLASS(dpose_recovery::DposeRecovery,
                       nav_core::RecoveryBehavior);
