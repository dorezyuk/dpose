#ifndef DPOSE_GOAL_TOLERANCE__DPOSE_GOAL_TOLERANCE__HPP
#define DPOSE_GOAL_TOLERANCE__DPOSE_GOAL_TOLERANCE__HPP

#include <dpose_core/dpose_core.hpp>

#include <gpp_interface/pre_planning_interface.hpp>

namespace dpose_goal_tolerance {

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
  dpose_core::gradient_decent opt_;
  double epsilon_ = 0;
  Map* map_ = nullptr;
};

}  // namespace dpose_goal_tolerance

#endif  // DPOSE_GOAL_TOLERANCE__DPOSE_GOAL_TOLERANCE__HPP
