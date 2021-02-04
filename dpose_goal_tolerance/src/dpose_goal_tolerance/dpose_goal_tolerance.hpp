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
#ifndef DPOSE_GOAL_TOLERANCE__DPOSE_GOAL_TOLERANCE__HPP
#define DPOSE_GOAL_TOLERANCE__DPOSE_GOAL_TOLERANCE__HPP

#include <dpose_core/dpose_core.hpp>
#include <dpose_core/dpose_costmap.hpp>
#include <dpose_goal_tolerance/DposeGoalToleranceConfig.h>

#include <gpp_interface/pre_planning_interface.hpp>
#include <costmap_2d/costmap_2d_ros.h>
#include <dynamic_reconfigure/server.h>

#include <IpIpoptApplication.hpp>
#include <IpTNLP.hpp>

#include <Eigen/Dense>

#include <memory>
#include <string>

namespace dpose_goal_tolerance {

/**
 * @brief Problem definition for pose cost optimization under constraints.
 *
 * The class defined the following ipopt-problem:
 * f = min(cost(x, y, theta))
 * g0: x^2 + y^2 < tol_lin^2
 * g1: theta^2 < tol_rot^2
 *
 * It searches for the displacements x to given initial pose p, such that the
 * cost of the displaced pose (p + x) is minimized. The displacement x is
 * constrained translationally (g0) and rotationally (g1).
 *
 */
struct problem : public Ipopt::TNLP {
  using index = Ipopt::Index;
  using number = Ipopt::Number;
  using pose = Eigen::Matrix<number, 3, 1>;
  using pose_gradient = dpose_core::pose_gradient;

private:
  using cell_vector = dpose_core::cell_vector;
  using costmap = costmap_2d::Costmap2DROS;

  // input vars
  number lin_tol_sq_ = 1;  ///< squared linear tolerance in cells
  number rot_tol_sq_ = 1;  ///< squared angular tolerance in rads
  pose pose_;              ///< initial pose in cells

  // cache
  number cost_;                ///< cost of the current solution
  pose_gradient::jacobian J_;  ///< jacobian of the current solution
  pose_gradient::hessian H_;   ///< hessian of the current solution

  // computation
  pose_gradient pg_;            ///< the pose-gradient object
  costmap *costmap_ = nullptr;  ///< pointer to a costmap
  cell_vector lethal_cells_;    ///< vector of lethal cells

  void
  on_new_x(index _n, const number *_x);

public:
  problem() = default;
  problem(costmap &_map, const pose_gradient::parameter &_param);

  void
  init(const pose &_pose, number _lin_tol, number _rot_tol);

  inline const pose &
  get_pose() const noexcept {
    return pose_;
  }

  bool
  get_nlp_info(index &_n, index &_m, index &_nonzero_jac_g,
               index &_nonzero_h_lag, IndexStyleEnum &_index_style) override;

  bool
  get_bounds_info(index _n, number *_x_lower, number *_x_upper, index _m,
                  number *_g_lower, number *_g_upper) override;
  bool
  get_starting_point(index _n, bool _init_x, number *_x, bool _init_z,
                     number *_z_L, number *_z_U, index _m, bool _init_lambda,
                     number *lambda) override;

  bool
  eval_f(index _n, const number *_x, bool _new_x, number &_f_value) override;

  bool
  eval_grad_f(index _n, const number *_x, bool _new_x,
              number *_grad_f) override;

  bool
  eval_g(index _n, const number *_x, bool _new_x, index _m,
         number *_g) override;

  bool
  eval_jac_g(index _n, const number *_x, bool _new_x, index _m,
             index _number_elem_jac, index *_i_row, index *_j_col,
             number *_jac_g) override;

  bool
  eval_h(index _n, const number *_x, bool _new_x, number _obj_factor, index _m,
         const number *_lambda, bool _new_lambda, index _number_elem_hess,
         index *_i_row, index *_j_col, number *_hess) override;

  void
  finalize_solution(Ipopt::SolverReturn _status, index _n, const number *_x,
                    const number *_z_L, const number *_z_U, index _m,
                    const number *_g, const number *_lambda, number _obj_value,
                    const Ipopt::IpoptData *_ip_data,
                    Ipopt::IpoptCalculatedQuantities *_ip_cq) override;
};

/**
 * @brief Class implements a goal-tolerance-functionality for global-planners.
 *
 * Add this to the pre_planning group of your GppPlugin in order to use it.
 * The class will **only** change the _goal (_start will remain unchanged).
 * You can define a list of tolerance types on the parameter server.
 *
 * See the README.md for more details.
 */
struct DposeGoalTolerance : public gpp_interface::PrePlanningInterface {
  bool
  preProcess(Pose &_start, Pose &_goal) override;

  void
  initialize(const std::string &_name, Map *_map) override;

private:
  // ipopt
  Ipopt::SmartPtr<Ipopt::TNLP> problem_;
  Ipopt::SmartPtr<Ipopt::IpoptApplication> solver_;

  // ros-comm
  Map *map_ = nullptr;
  ros::Publisher pose_pub_;

  // parameters
  double rot_tol_ = 1;
  double lin_tol_ = 1;

  // dynamic reconfigure
  using cfg_server = dynamic_reconfigure::Server<DposeGoalToleranceConfig>;
  using cfg_server_ptr = std::unique_ptr<cfg_server>;

  // shutdown the server first
  cfg_server_ptr cfg_server_;
};

}  // namespace dpose_goal_tolerance

#endif  // DPOSE_GOAL_TOLERANCE__DPOSE_GOAL_TOLERANCE__HPP
