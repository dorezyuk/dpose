#include <dpose_recovery/dpose_recovery.hpp>
#include <gtest/gtest.h>

#include <costmap_2d/layered_costmap.h>
#include <geometry_msgs/Point.h>
#include <tf2_ros/buffer.h>

#include <memory>
#include <vector>

using namespace dpose_recovery;
using geometry_msgs::Point;
using testing::Test;

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
    p_.reset(new problem{map_, {10, 2}, {5}});

    // mark obstacles in the costmap
    auto cm = map_.getCostmap();
    for (size_t ii = 0; ii < 200; ++ii) {
      cm->setCost(ii, 100, costmap_2d::LETHAL_OBSTACLE);
      cm->setCost(ii, 105, costmap_2d::LETHAL_OBSTACLE);
      cm->setCost(ii, 95, costmap_2d::LETHAL_OBSTACLE);
    }
  }

  costmap_2d::LayeredCostmap map_;
  std::unique_ptr<problem> p_;
};

TEST_F(costmap_fixture, init) {
  dpose_recovery::index n, m, nnz_jac_g, nnz_h_lag;
  Ipopt::TNLP::IndexStyleEnum index_style;
  ASSERT_TRUE(p_->get_nlp_info(n, m, nnz_jac_g, nnz_h_lag, index_style));
  ASSERT_EQ(n, 20);
  ASSERT_EQ(m, 0);
  ASSERT_EQ(nnz_jac_g, 0);
}

TEST_F(costmap_fixture, grad) {
  // init the u_vector
  const dpose_recovery::index N = 20;
  std::vector<dpose_recovery::number> u_vector(N, 0.1);
  std::vector<dpose_recovery::number> grad_f_vector(N);

  for (size_t ii = 0; ii < u_vector.size(); ii += 2)
    u_vector[ii] = 2;

  // set origin
  p_->set_origin(pose{100, 100, 0});

  // call the cost method
  dpose_recovery::number cost = 0;

  ASSERT_TRUE(p_->eval_f(N, u_vector.data(), true, cost));
  ASSERT_NE(cost, 0);

  // now get the gradient from the lib
  ASSERT_TRUE(p_->eval_grad_f(N, u_vector.data(), false, grad_f_vector.data()));

  // now compute the gradient manually: linear part
  dpose_recovery::number left_cost, right_cost;
  for (size_t ii = 0; ii < u_vector.size(); ++ii) {
    auto u_vector_disp = u_vector;
    u_vector_disp[ii] += 1e-6;
    ASSERT_TRUE(p_->eval_f(N, u_vector_disp.data(), true, right_cost));

    u_vector_disp[ii] -= 2e-6;
    ASSERT_TRUE(p_->eval_f(N, u_vector_disp.data(), true, left_cost));

    const auto grad = (right_cost - left_cost) / 2e-6;
    EXPECT_NEAR(grad, grad_f_vector[ii], 0.001) << "failed at " << ii;
  }
}

int
main(int argc, char** argv) {
  ros::init(argc, argv, "dpose_recovery");
  ros::NodeHandle nh;
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
