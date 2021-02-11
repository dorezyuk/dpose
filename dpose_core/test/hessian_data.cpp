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

  double mae = 0;  ///< mean abs error

  rotated_hessian() : se2(0, 0, GetParam()), pg(make_arrow(), {2, true}) {}
};

INSTANTIATE_TEST_SUITE_P(/**/, rotated_hessian, Range(0., 1.5, 0.1));

TEST_P(rotated_hessian, xx_grad) {
  for (size_t xx = 1; xx != 20; ++xx) {
    for (size_t yy = 0; yy != 20; ++yy) {
      // setup the cell-vector with the query
      cell_vector center_cells{cell(xx, yy)};
      cell_vector left_cells{cell(xx - 1, yy)};
      cell_vector right_cells{cell{xx + 1, yy}};

      // query the data
      pg.get_cost(se2, center_cells.cbegin(), center_cells.cend(), nullptr, &H);
      pg.get_cost(se2, left_cells.cbegin(), left_cells.cend(), &J_left,
                  nullptr);
      pg.get_cost(se2, right_cells.cbegin(), right_cells.cend(), &J_right,
                  nullptr);

      // compute the error
      const auto diff = (J_right.x() - J_left.x()) / 2.;
      const auto error = diff - H(0, 0);

      EXPECT_LE(std::abs(error), 0.4) << xx << ", " << yy;
      mae += std::abs(error / (diff ? diff : 1));
    }
  }
  mae /= (19 * 20);
  EXPECT_LE(mae, 0.41);
}

TEST_P(rotated_hessian, yy_grad) {
  for (size_t xx = 0; xx != 20; ++xx) {
    for (size_t yy = 1; yy != 20; ++yy) {
      cell_vector center_cells{cell(xx, yy)};
      cell_vector left_cells{cell(xx, yy - 1)};
      cell_vector right_cells{cell{xx, yy + 1}};

      pg.get_cost(se2, center_cells.cbegin(), center_cells.cend(), nullptr, &H);
      pg.get_cost(se2, left_cells.cbegin(), left_cells.cend(), &J_left,
                  nullptr);
      pg.get_cost(se2, right_cells.cbegin(), right_cells.cend(), &J_right,
                  nullptr);

      const auto diff = (J_right.y() - J_left.y()) / 2.;
      const auto error = diff - H(1, 1);

      EXPECT_LE(std::abs(error), 0.4) << xx << ", " << yy;
      mae += std::abs(error / (diff ? diff : 1));
    }
  }
  mae /= (19 * 20);
  EXPECT_LE(mae, 0.41);
}

}  // namespace