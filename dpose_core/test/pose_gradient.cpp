#include <dpose_core/dpose_core.hpp>
#include <gtest/gtest.h>

using namespace dpose_core;
using testing::Range;
using testing::TestWithParam;

namespace {
struct bad_footprint : public TestWithParam<int> {};

INSTANTIATE_TEST_SUITE_P(/**/, bad_footprint, Range(0, 2));

TEST_P(bad_footprint, generic) {
  // verifies that we react to footprints which don't define an area
  const polygon footprint(2, GetParam());
  EXPECT_ANY_THROW(pose_gradient(footprint, pose_gradient::parameter{}));
}

}  // namespace