#ifndef DPOSE_RECOVERY__DPOSE_RECVERY__HPP
#define DPOSE_RECOVERY__DPOSE_RECVERY__HPP

#include <dpose_core/dpose_core.hpp>
#include <dpose_core/dpose_costmap.hpp>

#include <nav_core/recovery_behavior.h>

#include <IpIpoptApplication.hpp>
#include <IpTNLP.hpp>

#include <Eigen/Dense>

#include <cmath>
#include <vector>

namespace dpose_recovery {

using number = Ipopt::Number;
using index = Ipopt::Index;
using dpose_core::pose_gradient;
using pose = pose_gradient::pose;

namespace diff_drive {

constexpr int x_dim = 3;
constexpr int u_dim = 2;

// where we pass u as input, and the initial x
using u_vector = Eigen::Matrix<number, u_dim, 1>;

inline pose
step(const pose &_x, const number &_u, const number &_w) noexcept {
  // _x is defined as [x, y, theta]^T
  // x = x + v * cos(theta)
  // y = y + v * sin(theta)
  // theta = theta + omega
  const pose x_d{std::cos(_x(2)) * _u, std::sin(_x(2)) * _u, _w};
  return _x + x_d;
}

using jacobian = Eigen::Matrix<number, x_dim, u_dim>;

inline void
T_jacobian(const number &_theta_prev, jacobian &_T_curr) noexcept {
  _T_curr << std::cos(_theta_prev), 0, std::sin(_theta_prev), 0, 0, 1;
}

// returns constant part of the Jacobian T_n based on the x_{n-1} state
inline void
T_jacobian(const pose &_x_prev, jacobian &_T_curr) noexcept {
  T_jacobian(_x_prev.z(), _T_curr);
}

inline void
R_jacobian(const number &_theta_prev, const number &_v_curr,
           jacobian &_R_curr) noexcept {
  _R_curr << 0, -_v_curr * std::sin(_theta_prev), 0,
      _v_curr * std::cos(_theta_prev), 0, 0;
}

inline void
R_jacobian(const pose &_x_prev, const u_vector &_u_curr,
           jacobian &_R_curr) noexcept {
  R_jacobian(_x_prev.z(), _u_curr.x(), _R_curr);
}

}  // namespace diff_drive

struct Problem : public Ipopt::TNLP {
  struct Parameter {
    size_t steps;  ///< number of look a steps in the horizon
    size_t dim_u = 2;
    size_t N;   ///< steps * dim_u
    double dt;  ///< time resolution, must be positive
    Eigen::Vector2d u_lower;
    Eigen::Vector2d u_upper;
  };

  Problem(costmap_2d::Costmap2DROS &_lcm, const Parameter &_our_param,
          const pose_gradient::parameter &_param);

  inline void
  set_origin(const Eigen::Vector3d &_x0) noexcept {
    x0_ = _x0;
    cost_best = std::numeric_limits<number>::max();
  }

  bool
  get_nlp_info(index &n, index &m, index &nnz_jac_g, index &nnz_h_lag,
               IndexStyleEnum &index_style) override;

  bool
  get_bounds_info(index n, number *x_l, number *x_u, index m, number *g_l,
                  number *g_u) override;

  bool
  get_starting_point(index n, bool init_x, number *x, bool init_z, number *z_L,
                     number *z_U, index m, bool init_lambda,
                     number *lambda) override;

  bool
  eval_f(index n, const number *x, bool new_x, number &obj_value) override;

  bool
  eval_grad_f(index n, const number *x, bool new_x, number *grad_f) override;

  bool
  eval_g(index n, const number *x, bool new_x, index m, number *g) override;

  bool
  eval_jac_g(index n, const number *x, bool new_x, index m, index nele_jac,
             index *iRow, index *jCol, number *values) override;

  void
  finalize_solution(Ipopt::SolverReturn status, index n, const number *x,
                    const number *z_L, const number *z_U, index m,
                    const number *g, const number *lambda, number obj_value,
                    const Ipopt::IpoptData *ip_data,
                    Ipopt::IpoptCalculatedQuantities *ip_cq) override;

  inline const std::vector<Eigen::Vector2d> &
  get_u() const noexcept {
    return u_best;
  }

  inline const number &
  get_cost() const noexcept {
    return cost_best;
  }

private:
  void
  on_new_u(index n, const number *u);

  std::vector<diff_drive::jacobian> T;
  std::vector<diff_drive::jacobian> R;
  std::vector<diff_drive::jacobian> R_hat;
  std::vector<pose_gradient::jacobian> J_hat;
  std::vector<Eigen::Matrix<number, 2, 1>> J_tilde;
  std::vector<Eigen::Vector2d> u_best;

  number cost;
  number cost_best;

  pose_gradient pg_;
  Eigen::Vector3d x0_;
  std::vector<Eigen::Vector3d> x_;
  costmap_2d::Costmap2DROS *map_ = nullptr;

  Parameter param_;
};

struct DposeRecovery : public nav_core::RecoveryBehavior {
  using Map = costmap_2d::Costmap2DROS;

  void
  initialize(std::string _name, tf2_ros::Buffer *_tf, Map *_global_map,
             Map *_local_map) override;

  void
  runBehavior() override;

private:
  tf2_ros::Buffer *tf_;
  Map *map_;
  Ipopt::SmartPtr<Ipopt::IpoptApplication> solver_;
  Ipopt::SmartPtr<Ipopt::TNLP> problem_;

  ros::Publisher pose_array_;
  ros::Publisher cmd_vel_;
};

}  // namespace dpose_recovery

#endif  // DPOSE_RECOVERY__DPOSE_RECVERY__HPP