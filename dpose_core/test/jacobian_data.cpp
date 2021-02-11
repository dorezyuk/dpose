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

  double mae = 0;  ///< mean abs error

  rotation() : se2(0, 0, GetParam()), pg(make_ship(), {10, false}) {}
};

INSTANTIATE_TEST_SUITE_P(/**/, rotation, Range(0., 1.5, 0.1));

TEST_P(rotation, x_grad) {
  double left_cost, right_cost;
  for (size_t xx = 1; xx != 20; ++xx) {
    for (size_t yy = 0; yy != 20; ++yy) {
      // setup the cell-vector with the query
      cell_vector center_cells{cell(xx, yy)};
      cell_vector left_cells{cell(xx - 1, yy)};
      cell_vector right_cells{cell{xx + 1, yy}};

      // query the data
      pg.get_cost(se2, center_cells.cbegin(), center_cells.cend(), &J, nullptr);
      left_cost = pg.get_cost(se2, left_cells.cbegin(), left_cells.cend(),
                              nullptr, nullptr);
      right_cost = pg.get_cost(se2, right_cells.cbegin(), right_cells.cend(),
                               nullptr, nullptr);

      // compute the error
      const auto diff = (right_cost - left_cost) / 2.;
      const auto error = diff - J.x();

      // we expect that we are "good enough". the value is rather high, since
      // there are still some discretesation issues.
      EXPECT_LE(std::abs(error), 0.1) << xx << ", " << yy;
      mae += std::abs(error / (diff ? diff : 1));
    }
  }
  mae /= (19 * 20);
  EXPECT_LE(mae, 0.06);
}

// copy and pasted from above - with the  y-values now under test
TEST_P(rotation, y_grad) {
  double lower_cost, upper_cost;
  for (size_t xx = 0; xx != 20; ++xx) {
    for (size_t yy = 1; yy != 20; ++yy) {
      cell_vector center_cells{cell(xx, yy)};
      cell_vector lower_cells{cell(xx, yy - 1)};
      cell_vector upper_cells{cell{xx, yy + 1}};

      pg.get_cost(se2, center_cells.cbegin(), center_cells.cend(), &J, nullptr);
      lower_cost = pg.get_cost(se2, lower_cells.cbegin(), lower_cells.cend(),
                               nullptr, nullptr);
      upper_cost = pg.get_cost(se2, upper_cells.cbegin(), upper_cells.cend(),
                               nullptr, nullptr);

      const auto diff = (upper_cost - lower_cost) / 2.;
      const auto error = diff - J.y();

      EXPECT_LE(std::abs(error), 0.1) << xx << ", " << yy;
      mae += std::abs(error / (diff ? diff : 1));
    }
  }
  mae /= (19 * 20);
  EXPECT_LE(mae, 0.05);
}

}  // namespace
