#include <dpose_goal_tolerance/dpose_goal_tolerance.hpp>
#include <gtest/gtest.h>

#include <costmap_2d/layered_costmap.h>
#include <geometry_msgs/Point.h>
#include <ros/ros.h>

#include <memory>
#include <vector>

using costmap_2d::LayeredCostmap;
using dpose_goal_tolerance::problem;
using geometry_msgs::Point;
using testing::Range;
using testing::Test;
using testing::WithParamInterface;

Point
make_point(double x, double y) noexcept {
  Point p;
  p.x = x;
  p.y = y;
  return p;
}

// fixture will setup a layered costmap, which we will use to generate some
// costs
struct costmap_fixture : public Test {
  costmap_fixture() : map_("map", false, false) {
    // resize the costmap so we can fit our footprint into it.
    map_.resizeMap(200, 200, 0.05, 0, 0, false);

    // create a footprint: a diamond
    const std::vector<Point> msg = {make_point(0.5, 0.0), make_point(0, 0.3),
                                    make_point(-0.2, 0.0), make_point(0, -0.3)};
    // set the footprint
    map_.setFootprint(msg);

    // create a new problem
    p_.reset(new problem{map_, {5}});

    // mark obstacles in the costmap
    auto cm = map_.getCostmap();
    for (size_t ii = 0; ii < 200; ++ii)
      cm->setCost(ii, 100, costmap_2d::LETHAL_OBSTACLE);
  }

  costmap_2d::LayeredCostmap map_;
  std::unique_ptr<problem> p_;
};

TEST_F(costmap_fixture, init) {
  dpose_goal_tolerance::index n, m, nnz_jac_g, nnz_h_lag;
  Ipopt::TNLP::IndexStyleEnum index_style;
  ASSERT_TRUE(p_->get_nlp_info(n, m, nnz_jac_g, nnz_h_lag, index_style));
  ASSERT_EQ(n, 3);
  ASSERT_EQ(m, 2);
  ASSERT_EQ(nnz_jac_g, 3);
}

// the parameter is yaw
struct rotated_costmap_fixture : public costmap_fixture,
                                 public WithParamInterface<double> {};

INSTANTIATE_TEST_SUITE_P(/**/, rotated_costmap_fixture, Range(0., 2.0, 0.1));

TEST_P(rotated_costmap_fixture, gradient) {
  // setup the start and goal poses
  dpose_goal_tolerance::pose start(99, 99, 0), goal(100, 100, GetParam());

  // setup the config
  dpose_goal_tolerance::DposeGoalToleranceConfig config;
  config.lin_tolerance = 1;
  config.rot_tolerance = 1;
  config.weight_goal_lin = 1;
  config.weight_goal_rot = 1;
  config.weight_start_lin = 1;
  config.weight_start_rot = 1;

  p_->init(start, goal, config);

  // get the info
  dpose_goal_tolerance::index n, m, nnz_jac_g, nnz_h_lag;
  Ipopt::TNLP::IndexStyleEnum index_style;
  ASSERT_TRUE(p_->get_nlp_info(n, m, nnz_jac_g, nnz_h_lag, index_style));

  // setup the displacement vector
  std::vector<dpose_goal_tolerance::number> x_vector(n, 0), grad_vector(n, 0);
  dpose_goal_tolerance::number cost, left_cost, right_cost;

  // get the cost
  ASSERT_TRUE(p_->eval_f(n, x_vector.data(), true, cost));

  // cost should not be zero
  ASSERT_NE(cost, 0);

  // get the gradient
  ASSERT_TRUE(p_->eval_grad_f(n, x_vector.data(), false, grad_vector.data()));

  // now compute the gradient yourself
  for (size_t ii = 0; ii != x_vector.size(); ++ii) {
    auto dx_vector = x_vector;
    dx_vector[ii] += 1e-6;
    ASSERT_TRUE(p_->eval_f(n, dx_vector.data(), true, right_cost));

    dx_vector[ii] -= 2e-6;
    ASSERT_TRUE(p_->eval_f(n, dx_vector.data(), true, left_cost));

    const auto grad = (right_cost - left_cost) / 2e-6;
    EXPECT_NEAR(grad, grad_vector[ii], 0.001) << "failed at " << ii;
  }
}

int
main(int argc, char** argv) {
  ros::init(argc, argv, "dpose_recovery");
  ros::NodeHandle nh;
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
