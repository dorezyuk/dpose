#include <dpose_core/dpose_core.hpp>
#include <gtest/gtest.h>

using namespace dpose_core;
using testing::TestWithParam;
using testing::Values;

namespace {

// parameter with a pose and the expected result
struct param {
  param(double x, double y, bool _ex) : pose(x, y, 0), expected(_ex) {}
  Eigen::Vector3f pose;
  bool expected;
};

// fixture with origin, bounds and paramters
struct fixture : public TestWithParam<param> {
  Eigen::Vector3f origin = Eigen::Vector3f::Zero();
  Eigen::Vector3f bounds = Eigen::Vector3f{2, 3, 0};
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