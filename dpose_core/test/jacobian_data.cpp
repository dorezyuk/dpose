#include <dpose_core/dpose_core.hpp>
#include <gtest/gtest.h>

#include <array>
#include <numeric>

using namespace dpose_core;
using testing::Range;
using testing::TestWithParam;

inline polygon
make_ship() {
  /*
   *  (-3, 2)    (4, 2)
   * (-6, 0) /--\
   *         \--/ (5, 0)
   * (-3, -2)    (4, -2)
   */
  polygon ship(2, 6);
  // clang-format off
    ship << 50,  40, -30, -60, -30, 40,
            0, -20, -20,  0,  20, 20;
  // clang-format on
  return ship;
}

namespace {

struct rotation : public TestWithParam<double> {};

INSTANTIATE_TEST_SUITE_P(/**/, rotation, Range(0., 1.5, 0.1));

TEST_P(rotation, x_grad) {
  pose_gradient::parameter param{3, false};
  pose_gradient pg(make_ship(), param);
  pose_gradient::pose se2(0, 0, GetParam());
  pose_gradient::jacobian J;

  // mean squared error
  double mse = 0;
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
      EXPECT_LE(error, 0.5) << xx << ", " << yy;
      mse += std::pow(error, 2);
    }
  }
  mse /= (19 * 20);
  EXPECT_LE(mse, 0.1);
}

// copy and pasted from above - with the  y-values now under test
TEST_P(rotation, y_grad) {
  pose_gradient::parameter param{3, false};
  pose_gradient pg(make_ship(), param);
  pose_gradient::pose se2(0, 0, GetParam());
  pose_gradient::jacobian J;

  // mean squared error
  double mse = 0;
  double left_cost, right_cost;

  for (size_t xx = 0; xx != 20; ++xx) {
    for (size_t yy = 1; yy != 20; ++yy) {
      // setup the cell-vector with the query
      cell_vector center_cells{cell(xx, yy)};
      cell_vector left_cells{cell(xx, yy - 1)};
      cell_vector right_cells{cell{xx, yy + 1}};

      // query the data
      pg.get_cost(se2, center_cells.cbegin(), center_cells.cend(), &J, nullptr);
      left_cost = pg.get_cost(se2, left_cells.cbegin(), left_cells.cend(),
                              nullptr, nullptr);
      right_cost = pg.get_cost(se2, right_cells.cbegin(), right_cells.cend(),
                               nullptr, nullptr);

      // compute the error
      const auto diff = (right_cost - left_cost) / 2.;
      const auto error = diff - J.y();

      // we expect that we are "good enough". the value is rather high, since
      // there are still some discretesation issues.
      EXPECT_LE(error, 0.5) << xx << ", " << yy;
      mse += std::pow(error, 2);
    }
  }
  mse /= (19 * 20);
  EXPECT_LE(mse, 0.1);
}

}  // namespace