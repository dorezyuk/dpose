#ifndef DPOSE_RECOVERY__DPOSE_RECVERY__HPP
#define DPOSE_RECOVERY__DPOSE_RECVERY__HPP

#include <dpose_core/dpose_core.hpp>

#include <nav_core/recovery_behavior.h>

#include <IpTNLP.hpp>
#include <Eigen/Dense>

#include <cmath>

namespace dpose_recovery {
namespace internal {

using namespace Ipopt;
using dpose_core::pose_gradient;

namespace diff_drive {
constexpr int x_dim = 3;
constexpr int u_dim = 2;

// where we pass u as input, and the initial x
using x_vector = Eigen::Matrix<Number, x_dim, 1>;
using u_vector = Eigen::Matrix<Number, u_dim, 1>;

inline x_vector
step(const x_vector &_x, const Number &_u, const Number &_w) noexcept {
  // _x is defined as [x, y, theta]^T
  // x = x + v * cos(theta)
  // y = y + v * sin(theta)
  // theta = theta + omega
  const x_vector x_d{std::cos(_x(2)) * _u, std::sin(_x(2)) * _u, _w};
  return _x + x_d;
}
}  // namespace diff_drive

struct pose_gradient_data {
  pose_gradient::jacobian J;
  pose_gradient::hessian H;
};


struct Problem : public TNLP {
  struct Parameter {
    size_t steps;  ///< number of look a steps in the horizon
    size_t dim_u = 2;
    size_t N;   ///< steps * dim_u
    double dt;  ///< time resolution, must be positive
    Eigen::Vector2d u_lower;
    Eigen::Vector2d u_upper;
    // double v_lower;
    // double v_upper;
    // double w_lower;
    // double w_upper;
  };

  Problem(costmap_2d::LayeredCostmap &_lcm, const Parameter &_our_param,
          const pose_gradient::parameter &_param);

  inline void
  set_origin(const Eigen::Vector3d &_x0) noexcept {
    x0_ = _x0;
  }

  bool
  get_nlp_info(Index &n, Index &m, Index &nnz_jac_g, Index &nnz_h_lag,
               IndexStyleEnum &index_style) override;

  bool
  get_bounds_info(Index n, Number *x_l, Number *x_u, Index m, Number *g_l,
                  Number *g_u) override;

  bool
  get_starting_point(Index n, bool init_x, Number *x, bool init_z, Number *z_L,
                     Number *z_U, Index m, bool init_lambda,
                     Number *lambda) override;

  bool
  eval_f(Index n, const Number *x, bool new_x, Number &obj_value) override;

  bool
  eval_grad_f(Index n, const Number *x, bool new_x, Number *grad_f) override;

  bool
  eval_g(Index n, const Number *x, bool new_x, Index m, Number *g) override;

  bool
  eval_jac_g(Index n, const Number *x, bool new_x, Index m, Index nele_jac,
             Index *iRow, Index *jCol, Number *values) override;

  bool
  eval_h(Index n, const Number *x, bool new_x, Number obj_factor, Index m,
         const Number *lambda, bool new_lambda, Index nele_hess, Index *iRow,
         Index *jCol, Number *values) override;

  void
  finalize_solution(SolverReturn status, Index n, const Number *x,
                    const Number *z_L, const Number *z_U, Index m,
                    const Number *g, const Number *lambda, Number obj_value,
                    const IpoptData *ip_data,
                    IpoptCalculatedQuantities *ip_cq) override;

private:
  pose_gradient pg_;
  Eigen::Vector3d x0_;
  std::vector<Eigen::Vector3d> x_;
  std::vector<pose_gradient_data> pg_data;

  Parameter param_;
};
}  // namespace internal

struct DposeRecovery : public nav_core::RecoveryBehavior {
  // way too long...
  using Map = costmap_2d::Costmap2DROS;

  void
  initialize(std::string _name, tf2_ros::Buffer *_tf, Map *_global_map,
             Map *_local_map) override;

  void
  runBehavior() override;

private:
  tf2_ros::Buffer *tf_;
  Map *map_;
};

}  // namespace dpose_recovery

#endif  // DPOSE_RECOVERY__DPOSE_RECVERY__HPP