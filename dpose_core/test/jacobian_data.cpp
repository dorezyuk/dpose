#include <dpose_core/dpose_core.hpp>
#include <gtest/gtest.h>

using namespace dpose_core;
using testing::Range;
using testing::TestWithParam;

inline polygon
make_ship() {
  /*
   *  (-30, 20)    (40, 20)
   * (-60, 0) /--\
   *          \--/ (50, 0)
   * (-30, -20)    (40, -20)
   */
  polygon ship(2, 6);
  // clang-format off
    ship << 50,  40, -30, -60, -30, 40,
            0, -20, -20,  0,  20, 20;
  // clang-format on
  return ship;
}

namespace {

struct rotation : public TestWithParam<double> {
  pose_gradient pg;
  pose_gradient::pose se2;
  pose_gradient::jacobian J;
  cell_vector cells;

  rotation() : se2(0, 0, GetParam()), pg(make_ship(), {10, false}) {
    // build the obstacle vector
    cells.reserve(20 * 20);
    for (size_t xx = 0; xx != 20; ++xx)
      for (size_t yy = 0; yy != 20; ++yy)
        cells.emplace_back(xx, yy);
  }
};

INSTANTIATE_TEST_SUITE_P(/**/, rotation, Range(0.1, 1.5, 0.1));

TEST_P(rotation, x_grad) {
  pose_gradient::pose offset(1e-6, 0, 0);
  pg.get_cost(se2, cells.cbegin(), cells.cend(), &J);

  // get the costs left and right of the pose
  const auto left_cost =
      pg.get_cost(se2 - offset, cells.cbegin(), cells.cend(), nullptr);
  const auto right_cost =
      pg.get_cost(se2 + offset, cells.cbegin(), cells.cend(), nullptr);

  // compute the relative error
  const auto diff = (right_cost - left_cost) / 2e-6;
  const auto error = std::abs((diff - J.x()) / (diff ? diff : 1.));

  // we expect that we are "good enough"
  EXPECT_LE(error, 0.001) << ": " << diff << " vs " << J.x();
}

// copy and pasted from above - with the  y-values now under test
TEST_P(rotation, y_grad) {
  pose_gradient::pose offset(0., 1e-6, 0);
  pg.get_cost(se2, cells.cbegin(), cells.cend(), &J);

  // get the costs left and right of the pose
  const auto lower_cost =
      pg.get_cost(se2 - offset, cells.cbegin(), cells.cend(), nullptr);
  const auto upper_cost =
      pg.get_cost(se2 + offset, cells.cbegin(), cells.cend(), nullptr);

  // compute the relative error
  const auto diff = (upper_cost - lower_cost) / 2e-6;
  const auto error = std::abs((diff - J.y()) / (diff ? diff : 1.));

  // we expect that we are "good enough"
  EXPECT_LE(error, 0.02) << ": " << diff << " vs " << J.y();
}

}  // namespace
