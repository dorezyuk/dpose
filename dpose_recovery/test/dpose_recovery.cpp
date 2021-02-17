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
    p_.reset(new problem{map_, {5, 2}, {5}});

    // mark obstacles in the costmap
    auto cm = map_.getCostmap();
    for (size_t ii = 0; ii < 200; ii += 20)
      cm->setCost(ii, 100, costmap_2d::LETHAL_OBSTACLE);
  }

  costmap_2d::LayeredCostmap map_;
  std::unique_ptr<problem> p_;
};

TEST_F(costmap_fixture, init) {
  dpose_recovery::index n, m, nnz_jac_g, nnz_h_lag;
  Ipopt::TNLP::IndexStyleEnum index_style;
  ASSERT_TRUE(p_->get_nlp_info(n, m, nnz_jac_g, nnz_h_lag, index_style));
  ASSERT_EQ(n, 10);
  ASSERT_EQ(m, 0);
  ASSERT_EQ(nnz_jac_g, 0);
}

TEST_F(costmap_fixture, grad) {
  // init the u_vector
  const dpose_recovery::index N = 10;
  std::vector<dpose_recovery::number> u_vector(N, 0.2);
  std::vector<dpose_recovery::number> grad_f_vector(N);

  // set origin
  p_->set_origin(pose{100, 100, 0});

  // call the cost method
  dpose_recovery::number cost = 0;

  ASSERT_TRUE(p_->eval_f(N, u_vector.data(), true, cost));
  ASSERT_NE(cost, 0);

  // now get the gradient from the lib
  ASSERT_TRUE(p_->eval_grad_f(N, u_vector.data(), false, grad_f_vector.data()));

  // the gradient should not be zero, since we want to do some math
  for(const auto& grad_f : grad_f_vector)
    ASSERT_NE(grad_f, 0);

  // now compute the gradient manually
}

int
main(int argc, char** argv) {
  ros::init(argc, argv, "dpose_recovery");
  ros::NodeHandle nh;
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
