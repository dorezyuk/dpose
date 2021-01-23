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

/**
 * @brief implements the MPC problem for avoiding obstacles.
 *
 * @note: you should run doxygen on this...
 *
 * @section Introduction
 *
 * The pose of the robot at the timestep \f$ n \f$ in a global frame is
 * given by \f$ \textbf{x}_n = \begin{bmatrix}x_{n} & y_{n} & \theta_{n}
 * \end{bmatrix}^T \f$.
 * Assuming a differential drive robot the control at the timestep \f$ n \f$ is
 * defined by \f$ \textbf{u}_{n} = \begin{bmatrix} v_{n} & \omega_{n}
 * \end{bmatrix}^T \f$. Here \f$ v_{n} \f$ denotes the translational and
 * \f$\omega_{n}\f$ the angular velocity. The kinematic model of the robot is
 * given by \f[ \begin{array}{rcl} \textbf{x}_{n} & = & \textbf{x}_{n-1} +
 * \begin{bmatrix}
 *                                                cos(\theta_{n-1}) & 0\\
 *                                                sin(\theta_{n-1}) & 0\\
 *                                                0 & 1
 *                                          \end{bmatrix} \textbf{u}_{n} \\
 *                & = & \textbf{x}_{n-1} + A_{n-1} \textbf{u}_{n}
 * \end{array}
 * \f]
 *
 * Due to the recursive nature of the equations above we can write every pose
 * \f$ \textbf{x}_{n} \f$ as a function of all previously issued controls \f$
 * \textbf{u}_{k}, k \le n \f$ and the start pose \f$ \textbf{x}_s \f$:
 * \f[
 * \textbf{x}_n = g(\textbf{x}_s, \textbf{u}_0, \textbf{u}_1, \dots,
 * \textbf{u}_n)
 * \f]
 *
 * We denote the cost of the robot at the pose \f$ \textbf{x}_{n} \f$ as \f$
 * f_{n}(\textbf{x}_n) \f$.
 * This cost represents the cost in the sense of the `costmap_2d` framework.
 * With the formula above we can write the cost \f$ f_n \f$ at the timestep \f$
 * n \f$ and the overall cost of the entire trajectory \f$ f \f$ as
 * \f[
 * \begin{array}{rcl}
 * f_{n} & = & f(g(\textbf{x}_s, \textbf{u}_0, \textbf{u}_1,
 * \dots,\textbf{u}_n)) \\ f & = & \sum_{n}^{N} f_{n} \end{array}
 * \f]
 *
 * We now formulate a MPC problem, where we want to find the sequence of
 * commands which minimizes the overall cost \f$ f \f$ of the trajectory:
 * \f[
 *  \min_{\textbf{u}} f(\textbf{u}) \\
 * \textbf{u}_{l} \le \textbf{u} \le \textbf{u}_u
 * \f]
 *
 *  Here \f$ \textbf{u}_l \f$ denotes the lower and \f$
 * \textbf{u}_u \f$ the upper limit for the command  \f$ \textbf{u} \f$.
 *
 * @section Solution
 *
 * `Ipopt` will mostly solve the problem for us.
 * All we need to provide is the jacobian and the hessian of the cost function
 * \f$ f \f$: \f[ \begin{array}{rcl}
 *
 * \nabla f & = & \begin{bmatrix}
 *              \frac{\partial f}{\partial v_{0}} &
 *              \frac{\partial f}{\partial \omega_{0}} &
 *              \cdots &
 *              \frac{\partial f}{\partial \omega_{N}}
 *            \end{bmatrix}^T \\
 * \nabla^2 f & = & \begin{bmatrix}
 *                  \frac{\partial f}{\partial v_{0} \partial v_{0}} &
 * \frac{\partial f}{\partial v_{0} \partial \omega_{0}} & \cdots \frac{\partial
 * f}{\partial v_{0} \partial \omega_{N}} \\
 *                  \frac{\partial f}{\partial \omega_{0} \partial v_{0}} &
 * \frac{\partial f}{\partial \omega_{0} \partial \omega_{0}} & \cdots
 * \frac{\partial f}{\partial \omega_{0} \partial \omega_{N}} \\
 *                  \vdots \\
 *                  \frac{\partial f}{\partial \omega_{N} \partial v_{0}} &
 * \frac{\partial f}{\partial \omega_{N} \partial \omega_{0}} & \cdots
 * \frac{\partial f}{\partial \omega_{N} \partial \omega_{N}}
 *
 *                  \end{bmatrix}
 * \end{array}
 *
 * \f]
 *
 * Every command \f$ \textbf{u}_n \f$ is an argument of the cost function \f$
 * f_m \f$ for \f$ n \le m \f$.
 * The derivative with respect to  \f$ \textbf{u}_n \f$ involves therefore the
 * cost terms \f$ f_m \f$ and is given by \f[ \frac{\partial f}{\partial
 * \textbf{u}_n} = \sum_{m=n}^{N}\frac{\partial f_{m}}{\partial \textbf{u}_n}
 * \f]
 *
 * The `dpose_core` library outputs the gradient of \f$ f_n \f$ with respect to
 * \f$ \textbf{x}_n \f$.
 * We can apply the chain-rule to get the desired data
 * \f[
 * \frac{\partial f_m}{\partial \textbf{u}_n} = \frac{\partial f_m}{\partial
 * \textbf{x}_m} \frac{\partial \textbf{x}_m}{\partial \textbf{u}_n}
 * \f]
 * The partial derivative of \f$ \textbf{x} \f$ with respect to \f$ \textbf{u}
 * \f$ is given by \f[ \frac{\partial \textbf{x}_m}{\partial \textbf{u}_n} =
 * \begin{bmatrix}
 *  -sin(\theta_{n-1}) & cos(\theta_{n-1}) \\
 *  cos(\theta_{n-1}) & -sin(\theta_{n-1}) \\
 *  0                    1
 * \end{bmatrix}
 * \f]
 */
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