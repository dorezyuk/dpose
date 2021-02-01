#include <dpose_core/dpose_core.hpp>
#include <gtest/gtest.h>

using namespace dpose_core;

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

TEST(jacobian_data, ship) {
  pose_gradient::parameter param{3, false};
  pose_gradient pg(make_ship(), param);

  // setup the cell-vector with the query
  cell_vector cells;

  // setup the pose with 45 deg
  pose_gradient::pose se2(0, 0, M_PI / 4.);
}