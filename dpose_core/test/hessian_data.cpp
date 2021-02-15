#include <dpose_core/dpose_core.hpp>
#include <gtest/gtest.h>

using namespace dpose_core;
using testing::Range;
using testing::TestWithParam;

inline polygon
make_arrow() {
  /*
   *  (-60, 20)    (40, 20)
   * (-30, 0) \--\
   *          /--/ (50, 0)
   * (-60, -20)    (40, -20)
   */
  polygon arrow(2, 6);
  // clang-format off
    arrow << 50,  40, -60, -30, -60, 40,
             0, -20, -20,  0,  20, 20;
  // clang-format on
  return arrow;
}

namespace {

struct rotated_hessian : public TestWithParam<double> {
  pose_gradient pg;
  pose_gradient::pose se2;
  pose_gradient::jacobian J_left, J_right;
  pose_gradient::hessian H;
  cell_vector cells;

  double mae = 0;  ///< mean abs error

  rotated_hessian() : se2(0, 0, GetParam()), pg(make_arrow(), {2, true}) {
    cells.reserve(20 * 20);
    for (size_t xx = 0; xx != 20; ++xx)
      for (size_t yy = 0; yy != 20; ++yy)
        cells.emplace_back(xx, yy);
  }
};

INSTANTIATE_TEST_SUITE_P(/**/, rotated_hessian, Range(0., 1.5, 0.1));

TEST_P(rotated_hessian, DISABLED_xx_grad) {
  pose_gradient::pose offset(0.01, 0, 0);
  pg.get_cost(se2, cells.cbegin(), cells.cend(), nullptr, &H);

  // get the Jacobians left and right of the pose
  pg.get_cost(se2 - offset, cells.cbegin(), cells.cend(), &J_left, nullptr);
  pg.get_cost(se2 + offset, cells.cbegin(), cells.cend(), &J_right, nullptr);

  // compute the relative error
  const auto diff = (J_left.x() - J_right.x()) / 0.02;
  const auto error = std::abs((diff - H(0, 0)) / (diff ? diff : 1.));

  // we expect that we are "good enough"
  EXPECT_LE(error, 0.07) << ": " << diff << " vs " << H(0, 0);
}

TEST_P(rotated_hessian, DISABLED_yy_grad) {
  pose_gradient::pose offset(0, 0.01, 0);
  pg.get_cost(se2, cells.cbegin(), cells.cend(), nullptr, &H);

  // get the Jacobians lower and upper to the pose
  pg.get_cost(se2 - offset, cells.cbegin(), cells.cend(), &J_left, nullptr);
  pg.get_cost(se2 + offset, cells.cbegin(), cells.cend(), &J_right, nullptr);

  // compute the relative error
  const auto diff = (J_left.y() - J_right.y()) / 0.02;
  const auto error = std::abs((diff - H(1, 1)) / (diff ? diff : 1.));

  // we expect that we are "good enough"
  EXPECT_LE(error, 0.12) << ": " << diff << " vs " << H(1, 1);
}

}  // namespace