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

#include <gpp_interface/pre_planning_interface.hpp>
#include <angles/angles.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

namespace dpose_goal_tolerance {
namespace internal {

/**
 * @brief class for tolerance checks.
 *
 * Given the points _a and _b, this class will check if _a is within a
 * tolerance of _b.
 *
 * Pass one or multiple modes (tolerance::mode) to the c'tor, in order to
 * determine the "within" condition: the point _a is within the tolerance of _b
 * if all modes return true.
 */
struct tolerance {
  /// @brief diffent "modes"
  enum class mode { NONE, ANGLE, SPHERE, BOX };

  using pose = Eigen::Vector3d;

  /**
   * @brief noop-tolerance.
   *
   * All other tolerance concepts will extend this class.
   * We will use dynmic polymorphism in order to implement different concepts.
   */
  struct none_tolerance {
    virtual ~none_tolerance() = default;

    inline virtual bool
    within(const pose& _a __attribute__((unused)),
           const pose& _b __attribute__((unused))) const noexcept {
      return true;
    }
  };

  /**
   * @brief angle_tolerance
   *
   * interprets the last value of pose as a angle (in rads) and checks if the
   * angular distance from _a to _b is within the tolerance.
   */
  struct angle_tolerance : public none_tolerance {
    explicit angle_tolerance(double _tol);

    inline bool
    within(const pose& _a, const pose& _b) const noexcept final {
      return std::abs(angles::shortest_angular_distance(_a.z(), _b.z())) < tol_;
    }

  private:
    double tol_;  ///< tolerance in radians
  };

  /**
   * @brief tolerance on a sphere
   *
   * The point _a is within the tolerance to _b, if their normed difference is
   * below the radius_ parameter.
   *
   * @note: radius_ will be redrived as _norm.norm() from the
   * tolerance::tolerance call.
   */
  struct sphere_tolerance : public none_tolerance {
    explicit sphere_tolerance(const pose& _norm);

    inline bool
    within(const pose& _a, const pose& _b) const noexcept final {
      return (_a - _b).norm() <= radius_;
    }

  private:
    double radius_;  ///< radius in meters
  };

  /**
   * @brief tolerance on a box
   *
   * The point _a is within the tolerance of _b if _b fits in a box of the size
   * box_ * 2 centered around _a.
   */
  struct box_tolerance : public none_tolerance {
    explicit box_tolerance(const pose& _box);

    inline bool
    within(const pose& _a, const pose& _b) const noexcept final {
      return (((_a - _b).array().abs() - box_.array()) <= 0).all();
    }

  private:
    pose box_;  ///< half size of the box
  };

private:
  using tolerance_ptr = std::unique_ptr<none_tolerance>;
  std::vector<tolerance_ptr> impl_;

  static tolerance_ptr
  factory(const mode& _m, const pose& _center) noexcept;

public:
  tolerance() = default;
  tolerance(const mode& _m, const pose& _center);

  // define the list-type. for now we dont want to use the
  // std::initializer_list, since we want to be able to dynamically allocate the
  // list.
  using pair_type = std::pair<mode, pose>;
  using list_type = std::vector<pair_type>;
  tolerance(const list_type& _list);

  inline bool
  within(const pose& _a, const pose& _b) const noexcept {
    // check all tolerances defined
    return std::all_of(
        impl_.begin(), impl_.end(),
        [&](const tolerance_ptr& _impl) { return _impl->within(_a, _b); });
  }
};

using namespace dpose_core;

/// @brief returns all lethal cells within the _bounds
/// @param _map the costmap to search
/// @param _bounds the bounds of the search
cell_vector
lethal_cells_within(const costmap_2d::Costmap2D& _map,
                    const rectangle<int>& _bounds);

/**
 * @brief gradient decent optimizer for the pose_gradient.
 *
 * Will perform the gradient decent until a termination condition is met.
 * The decent ends
 * - if either the maximum iterations are reached,
 * - if the cost lies below the epsilon bound,
 * - if the derived position lies outside of the tolerance tol w.r.t _start.
 */
struct gradient_decent {
  /// @brief parameter for the optimization
  struct parameter {
    size_t iter = 10;      ///< maximal number of steps
    double step_t = 1;     ///< maximal step size for translation (in cells)
    double step_r = 0.1;   ///< maximal step size for rotation (in rads)
    double epsilon = 0.1;  ///< cost-threshold for termination
    tolerance tol;         ///< maximal distance from _start
  };

  gradient_decent() = default;
  explicit gradient_decent(parameter&& _param) noexcept;

  std::pair<float, Eigen::Vector3d>
  solve(const pose_gradient& _pg, const pose_gradient::pose& _start,
        const cell_vector& _cells) const;

  inline const parameter&
  get_param() const noexcept {
    return param_;
  }

private:
  parameter param_;  ///< parameterization for the optimization
};

}  // namespace internal

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
  preProcess(Pose& _start, Pose& _goal) override;

  void
  initialize(const std::string& _name, Map* _map) override;

private:
  dpose_core::pose_gradient grad_;
  internal::gradient_decent opt_;
  double epsilon_ = 0;
  Map* map_ = nullptr;
};

}  // namespace dpose_goal_tolerance

#endif  // DPOSE_GOAL_TOLERANCE__DPOSE_GOAL_TOLERANCE__HPP
