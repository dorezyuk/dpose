#include <dpose_core/dpose_costmap.hpp>
#include <gtest/gtest.h>

#include <costmap_2d/costmap_2d.h>

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

using dpose_core::cell;
using dpose_core::check_footprint;
using dpose_core::is_inside;
using dpose_core::polygon;
using dpose_core::raytrace;
using dpose_core::x_bresenham;
using testing::Range;
using testing::Test;
using testing::TestWithParam;
using testing::ValuesIn;
using testing::WithParamInterface;

namespace test_is_inside {
// contains the tests for dpose_core::is_inside function

/// @brief fixture which provides us with a costmap
struct is_inside_fixture : public Test {
  costmap_2d::Costmap2D cm_;
  is_inside_fixture() : cm_(30, 40, 0.1, 0, 0) {}
};

// simple check that we handle empty polygons correctly
TEST_F(is_inside_fixture, empty) { EXPECT_TRUE(is_inside(cm_, polygon{})); }

/// @brief extends the base fixture by providing a polygon which has the exact
/// size of the costmap.
struct is_inside_fixture_with_poly : public is_inside_fixture {
  polygon footprint;
  is_inside_fixture_with_poly() : footprint(2, 4) {
    const auto x = cm_.getSizeInCellsX() - 1, y = cm_.getSizeInCellsY() - 1;
    // clang-format off
    footprint << 0, x, x, 0,
                 0, 0, y, y;
    // clang-format on
  }
};

// checks that we handle the edge case (footprint fits exactly in the costmap)
// correclty
TEST_F(is_inside_fixture_with_poly, fits) {
  EXPECT_TRUE(is_inside(cm_, footprint));
}

/// @brief the parameter's first is the index of the footprint. it's second is
/// the perturbation we apply.
using is_outside_param = std::pair<Eigen::Index, int>;
struct is_outside_fixture : public is_inside_fixture_with_poly,
                            public WithParamInterface<is_outside_param> {};

// the array with the applied disturbances (index and offset).
is_outside_param params[] = {{0, -1}, {1, -1}, {2, 1},  {3, -1},
                             {4, 1},  {5, 1},  {6, -1}, {7, 1}};

INSTANTIATE_TEST_SUITE_P(/**/, is_outside_fixture, ValuesIn(params));

// we now apply a small offset to the polygon, of which we know that it fits
// perfectly in our costmap. with the offset we expect it to be outside of the
// costmap bounds.
TEST_P(is_outside_fixture, generic) {
  const auto param = GetParam();
  footprint(param.first) += param.second;
  EXPECT_FALSE(is_inside(cm_, footprint));
}

}  // namespace test_is_inside

namespace test_x_bresenham {

/// @brief will construct a rotated line. the parameter is the angle of the
/// line.
struct rotated_line_fixture : public TestWithParam<double> {
  const cell zero;            ///< origin of the line: we will keep it fixed
  cell end;                   ///< the rotated end of the line
  const double radius = 100;  ///< the length of our line

  rotated_line_fixture() : zero(0, 0) {}
};

// we just check the upper half of the circle, so our zero remains the lowert
// point.
INSTANTIATE_TEST_SUITE_P(/**/, rotated_line_fixture, Range(0.1, 1.4, 0.1));

// test compares the x-bresenham to the raytrace function.
TEST_P(rotated_line_fixture, generic) {
  // generate the end
  end << std::cos(GetParam()) * radius, std::sin(GetParam()) * radius;

  // check the correct setup of the test: end's y must be positive
  ASSERT_GT(end.y(), zero.y());

  // create our x-bresenham instance
  x_bresenham br(zero, end);

  // create the default ray
  const auto ray = raytrace(zero, end);

  // now we iterate over the ray and compare the values to our x_bresenham
  // output
  auto curr_y = -1;
  for (const auto& point : ray) {
    // we have a new y: check the output
    if (point.y() != curr_y) {
      EXPECT_EQ(point.x(), br.get_x());
      br.advance_x();
      curr_y = point.y();
    }
  }
}

// here we test that the output of x_bresenham always starts with the lowest
// y-value
TEST(test_x_bresenham, order_invariance) {
  // generate the end
  cell end(10, 12), zero(0, 0);

  // create two instances with swapped arguments orders
  x_bresenham up(zero, end), down(end, zero);

  // check the start
  EXPECT_EQ(up.get_x(), down.get_x());
}

}  // namespace test_x_bresenham

namespace test_check_footprint {

// very artistic!
inline polygon
make_hand() {
  polygon hand(2, 19);
  // clang-format off
  hand << 20, 20, 10, 10, 30, 30, 11, 11, 32, 32, 11, 11, 31, 31,  5, 10, 9,   -5, -6,
          12, 11, 10,  8,  8,  6,  6,  3,  2, -1, -2, -4, -5, -7, -6, -10,-11, -6,  9;
  // clang-format on
  return hand;
}

inline polygon
make_square() {
  polygon square(2, 4);
  square << 10, 10, -12, -5, -10, 8, 9, -7;
  return square;
}

// the parameter is yaw of the footprint
struct check_footprint_fixture : public TestWithParam<double> {
  costmap_2d::Costmap2D cm_;
  polygon footprint_;

  check_footprint_fixture() :
      cm_(100, 100, 0.1, 0, 0), footprint_(make_square()) {}
};

int counter = 0;

INSTANTIATE_TEST_SUITE_P(/**/, check_footprint_fixture,
                         Range(0., M_PI * 2, 0.1));

// we test our implementation on the slower ros's checks
TEST_P(check_footprint_fixture, regression) {
  // move the footprint into the middle
  const Eigen::Translation2d trans(cm_.getSizeInCellsX() / 2,
                                   cm_.getSizeInCellsY() / 2);
  const Eigen::Rotation2Dd rot(GetParam());
  const Eigen::Isometry2d iso = trans * rot;

  // create the shifted footprint
  const polygon shifted_fp =
      (iso * footprint_.cast<double>()).array().round().cast<int>();

  // now generate the ros-footprint
  std::vector<costmap_2d::MapLocation> vertices, outline, area;
  vertices.resize(shifted_fp.cols());

  for (size_t ii = 0; ii != vertices.size(); ++ii)
    vertices.at(ii) =
        costmap_2d::MapLocation{shifted_fp(0, ii), shifted_fp(1, ii)};

  cm_.polygonOutlineCells(vertices, outline);
  cm_.convexFillCells(outline, area);

  // empty would not make sense here.
  ASSERT_FALSE(area.empty());

  // we now mark everything outside of the footprint cells as occupied
  auto raw_costmap = cm_.getCharMap();
  const auto map_size = cm_.getSizeInCellsX() * cm_.getSizeInCellsY();
  std::fill_n(raw_costmap, map_size, costmap_2d::LETHAL_OBSTACLE);

  for (const auto& cell : area)
    cm_.setCost(cell.x, cell.y, costmap_2d::FREE_SPACE);

  // we now check if our cost-check returns free
  EXPECT_TRUE(check_footprint(cm_, shifted_fp, costmap_2d::LETHAL_OBSTACLE));

  for (const auto& cell : outline) {
    cm_.setCost(cell.x, cell.y, costmap_2d::LETHAL_OBSTACLE);
    EXPECT_FALSE(check_footprint(cm_, shifted_fp, costmap_2d::LETHAL_OBSTACLE))
        << cell.x << ", " << cell.y;
    cm_.setCost(cell.x, cell.y, costmap_2d::FREE_SPACE);
  }
}

}  // namespace test_check_footprint
