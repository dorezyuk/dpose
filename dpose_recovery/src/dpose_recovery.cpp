#include <dpose_recovery/dpose_recovery.hpp>

#include <cassert>

namespace dpose_recovery {
namespace internal {

Problem::Problem(costmap_2d::LayeredCostmap &_lcm, const Parameter &_our_param,
                 const pose_gradient::parameter &_param) :
    pg_(_lcm, _param),
    param_(_our_param),
    x_(_our_param.steps),
    pg_data(_our_param.steps) {
  // todo not sure about this lets see
  // if (!param_.steps)
  // throw std::invalid_argument("steps cannot be zero");
}

bool
Problem::get_nlp_info(Index &n, Index &m, Index &nnz_jac_g, Index &nnz_h_lag,
                      IndexStyleEnum &index_style) {
  // number of variables
  // dim_u should depend on the robot model (diff-drive vs omni-drive)
  n = param_.steps * param_.dim_u;

  // one constraint: the cost must be non-lethal
  m = 0;

  // todo this is not quite true
  nnz_jac_g = 0;
  nnz_h_lag = 10;

  index_style = TNLP::C_STYLE;
  return true;
};

bool
Problem::get_bounds_info(Index n, Number *x_l, Number *x_u, Index m,
                         Number *g_l, Number *g_u) {
  assert(n == param_.steps * param_.dim_u);
  assert(m == 1);

  for (Index ii = 0; ii != n; ++ii) {
    x_l[ii] = param_.v_lower;
    x_l[++ii] = param_.w_lower;
  }

  for (Index ii = 0; ii != n; ++ii) {
    x_u[ii] = param_.v_upper;
    x_u[++ii] = param_.w_upper;
  }

  return true;
};

bool
Problem::get_starting_point(Index n, bool init_x, Number *x, bool init_z,
                            Number *z_L, Number *z_U, Index m, bool init_lambda,
                            Number *lambda) {
  assert(init_x == true);
  assert(init_z == false);
  assert(init_lambda == false);

  // todo initialize to the given starting point
  // todo what a good guess, how can I hot-start and when to reset?
  return true;
};

bool
Problem::eval_f(Index n, const Number *u, bool new_u, Number &cost) {
  assert(n == param_.steps * param_.dim_u);
  // get the x-states based on the u commands: x_curr = f(x_prev, u_curr)
  // x_1 = x_0 + v_1 * cos(z_0)
  // y_1 = y_0 + v_1 * sin(z_0)
  // z_1 = z_0 + w_1
  // init the first step
  x_[0] = diff_drive::step(x0_, u[0], u[1]);

  // jj is the index within u, which has the form [v0, w0, v1, w1...]
  for (size_t ii = 1, jj = 2; ii != param_.steps; ++ii, jj += 2)
    x_[ii] = diff_drive::step(x_[ii - 1], u[jj], u[jj + 1]);

  // get the sum of the costs, J and H for every state
  cost = 0;
  for (size_t ii = 0; ii != param_.steps; ++ii)
    cost += pg_.get_cost(x_[ii].cast<float>(), &pg_data[ii].J, &pg_data[ii].H);

  return true;
}

bool
Problem::eval_grad_f(Index n, const Number *u, bool new_u, Number *grad_f) {
  assert(n == param_.steps * param_.dim_u);

  if (new_u) {
    // todo init the pg_data again...
  }

  // get the gradient for every u; the final result will look like
  // [df / du_0, df / dw_0, df / du_1, df / dw_1 ..., df / du_N, df / dw_N]
  float theta = x0_.z();
  // get the sum of all thetas
  for (size_t ii = 0; ii != param_.steps; ++ii)
    theta += u[ii * 2 + 1];

  pose_gradient::jacobian J = pose_gradient::jacobian::Zero();
  for (int ii = param_.steps, nn = n; ii != 0; --ii, --nn) {
    // dpose_core returns df_n / dx_n. we apply the chain rule to get
    // df / du_i = sum over n((df_n / dx_n) * (dx_n / du_i)).
    // dx_n / du_i is zero if n < i, since earlier steps are independent from
    // their successors. for n >= i following applies:
    // d_x_n / d_v_i = cos(z_{0...n-1})
    // d_y_n / d_v_i = sin(z_{0...n-1})
    // d_z_n / d_v_1 = 0
    // d_x_1 / d_w_1 = -sin(z_0)
    // d_y_1 / d_w_1 = cos(z_0)
    // d_z_1 / d_w_1 = 1
    // accumulate the J
    J += pg_data[ii - 1].J;

    // get the transformation matrix A
    const auto cos_theta = std::cos(theta);
    const auto sin_theta = std::sin(theta);
    theta -= u[nn];

    Eigen::Matrix<float, 3UL, 2UL> A;
    // clang-format off
    A << cos_theta, -sin_theta,
         sin_theta,  cos_theta,
         0,          1;
    // clang-format on

    const Eigen::Matrix<float, 2UL, 1UL> grad_fn = A.transpose() * J;
    grad_f[nn] = grad_fn(1);
    grad_f[--nn] = grad_fn(0);
  }
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
  assert(n == param_.steps * param_.dim_u);
  if (new_u) {
    // init the pg-data
  }

  if (!values) {
    // init the rows and cols
    // [[df / (du0 du0), df / (du0 dw0), ..., df / (du0 dwN)],
    //  [df / (dw0 du0), df / (dw0 dw0), ..., df / (dw0 dwN)],
    //  ...
    //  [df / (dwN du0), df / (dwN dw0), ..., df / (dwN dwN)]]
    Index idx = 0;
    for (Index row = 0; row != n; row++)
      for (Index col = 0; col <= row; col++) {
        iRow[idx] = row;
        jCol[idx] = col;
        idx++;
      }

    assert(idx == n);
  }
  else {
    // populate the hessian
    float theta = x0_.z();
    // get the sum of all thetas
    for (size_t ii = 0; ii != param_.steps; ++ii)
      theta += u[ii * 2 + 1];

    pose_gradient::hessian H = pose_gradient::hessian::Zero();
    for (int ii = param_.steps, nn = n; ii != 0; --ii, --nn) {
      // accumulate the H
      H += pg_data[ii - 1].H;

      // get the transformation matrix A
      const auto cos_theta = std::cos(theta);
      const auto sin_theta = std::sin(theta);
      theta -= u[nn];

      Eigen::Matrix<float, 3UL, 2UL> A;
      // clang-format off
      A << cos_theta, -sin_theta,
           sin_theta,  cos_theta,
           0,          1;
      // clang-format on


      const Eigen::Matrix<float, 2UL, 2UL> H_ii = A.transpose() * H * A;
    }
  }
  return true;
}

void
Problem::finalize_solution(SolverReturn status, Index n, const Number *x,
                           const Number *z_L, const Number *z_U, Index m,
                           const Number *g, const Number *lambda,
                           Number obj_value, const IpoptData *ip_data,
                           IpoptCalculatedQuantities *ip_cq){};
}  // namespace internal

void
DposeRecovery::initialize(std::string _name, tf2_ros::Buffer *_tf,
                          Map *_global_map, Map *_local_map) {
  tf_ = _tf;
  map_ = _local_map;
}

void
DposeRecovery::runBehavior() {}
}  // namespace dpose_recovery
