#include <dpose_core/dpose_core.hpp>
#include <gtest/gtest.h>

using namespace dpose_core;
using testing::TestWithParam;
using testing::Values;

namespace {

// parameter with a pose and the expected result
struct param {
  param(double x, double y, bool _ex) : pose(x, y), expected(_ex) {}
  Eigen::Vector2d pose;
  bool expected;
};

// fixture with origin, bounds and paramters
struct fixture : public TestWithParam<param> {
  Eigen::Vector2d origin = Eigen::Vector2d::Zero();
  Eigen::Vector2d bounds = Eigen::Vector2d{2, 3};
  param p;
  fixture() : p(GetParam()) {}
};

using spheric_fixture = fixture;
using box_fixture = fixture;
INSTANTIATE_TEST_SUITE_P(spheric, spheric_fixture,
                         Values(param{2, 3, true}, param{3, 3, false},
                                param{-3, -2, true}, param{-3, -3, false}));

TEST_P(spheric_fixture, generic) {
  const tolerance t(tolerance::mode::SPHERE, bounds);
  EXPECT_EQ(t.within(origin, p.pose), p.expected);
}

INSTANTIATE_TEST_SUITE_P(box, box_fixture,
                         Values(param{2, 3, true}, param{3, 2, false},
                                param{-2, -3, true}, param{-3, -2, false}));

TEST_P(box_fixture, generic) {
  const tolerance t(tolerance::mode::BOX, bounds);
  EXPECT_EQ(t.within(origin, p.pose), p.expected);
}

}  // namespace