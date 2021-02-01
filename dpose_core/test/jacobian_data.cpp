#include <dpose_core/dpose_core.hpp>
#include <gtest/gtest.h>

using namespace dpose_core;
using testing::TestWithParam;
using testing::Range;

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
    ship << 5,  4, -3, -6, -3, 4,
            0, -2, -2,  0,  2, 2;
  // clang-format on
  return ship;
}

namespace {

struct rotation : public TestWithParam<double> {};

INSTANTIATE_TEST_SUITE_P(/**/, rotation, Range(0., 1.5, 0.1));

TEST_P(rotation, x_grad) {
  pose_gradient::parameter param{3, false};
  pose_gradient pg(make_ship(), param);

  // we will swipe through yy
  for (size_t yy = 0; yy != 10; ++yy) {
    // setup the cell-vector with the query
    cell_vector center_cells{cell(1, yy)};
    cell_vector left_cells{cell(0, yy)};
    cell_vector right_cells{cell{2, yy}};
    // std::cout << yy << std::endl;

    // setup the pose with 45 deg
    pose_gradient::pose se2(0, 0, GetParam());
    pose_gradient::jacobian J;

    pg.get_cost(se2, center_cells.cbegin(), center_cells.cend(), &J, nullptr);
    auto left_cost = pg.get_cost(se2, left_cells.cbegin(), left_cells.cend(),
                                 nullptr, nullptr);
    auto right_cost = pg.get_cost(se2, right_cells.cbegin(), right_cells.cend(),
                                  nullptr, nullptr);

    // std::cout << "left " << left_cost << std::endl;
    // std::cout << "right " << right_cost << std::endl;
    // std::cout << J.transpose() << std::endl;
    const auto diff = right_cost - left_cost;
    std::cout << diff - J.x() << std::endl;
    // std::cout << "------------------------------" << std::endl;
  }
}
}
