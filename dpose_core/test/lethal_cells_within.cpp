#include <dpose_core/dpose_costmap.hpp>
#include <gtest/gtest.h>

#include <costmap_2d/cost_values.h>
#include <costmap_2d/costmap_2d.h>

#include <algorithm>
#include <set>
#include <vector>

using costmap_2d::Costmap2D;
using costmap_2d::LETHAL_OBSTACLE;
using costmap_2d::MapLocation;
using Eigen::Matrix2d;
using Eigen::Rotation2Dd;
using Eigen::Vector2i;
using testing::Range;
using testing::Test;
using testing::WithParamInterface;

using namespace dpose_core;

// setups a costmap, so we can run some tests on it
struct costmap_fixture : public Test {
  costmap_fixture() : map_(200, 200, 0.1, 0, 0) {}
  Costmap2D map_;
};

TEST_F(costmap_fixture, dot) {
  // paint a dot into the costmap
  map_.setCost(100, 100, LETHAL_OBSTACLE);

  // get the box
  rectangle<int> box = to_rectangle(Vector2i{99, 99}, Vector2i{101, 101});

  // get the cells
  const auto cells = lethal_cells_within(map_, box);

  // check the result
  ASSERT_EQ(cells.size(), 1);
  ASSERT_EQ(cells.front(), cell(100, 100));
}

TEST_F(costmap_fixture, line) {
  // paint a sparse line into the costmap
  for (size_t ii = 0; ii < map_.getSizeInCellsX(); ii += 20)
    map_.setCost(ii, 100, LETHAL_OBSTACLE);

  // get the box
  rectangle<int> box = to_rectangle(Vector2i{80, 99}, Vector2i{121, 101});

  // get the cells
  const auto cells = lethal_cells_within(map_, box);

  // check the output
  ASSERT_EQ(cells.size(), 3);
}

// the next tests are regression tests against the vanilla ros-implementation.
// the parameter is the rotation angle
struct lethal_costmap_fixture : public costmap_fixture,
                                public WithParamInterface<double> {
  lethal_costmap_fixture() :
      box_(to_rectangle(Vector2i{-20, -10}, Vector2i{30, 20})) {
    const auto cells = map_.getSizeInCellsX() * map_.getSizeInCellsY();
    auto m = map_.getCharMap();

    // paint it... lethal
    std::fill_n(m, cells, LETHAL_OBSTACLE);
  }

  const rectangle<int> box_;  ///< bounding box to check
};

INSTANTIATE_TEST_SUITE_P(/**/, lethal_costmap_fixture, Range(0., 2., 0.1));

TEST_P(lethal_costmap_fixture, regression) {
  // get the rotated bounding box
  Eigen::Isometry2d rot =
      Eigen::Translation2d(100, 100) * Rotation2Dd(GetParam());
  rectangle<int> rot_box =
      (rot * box_.cast<double>()).array().round().cast<int>().matrix();

  // get the lethal cells
  const auto cells = lethal_cells_within(map_, rot_box);

  // quick check
  ASSERT_FALSE(cells.empty());

  // get the lethal cells with the ros-impl
  // create the outline in ros cells
  std::vector<MapLocation> outline, area;
  outline.reserve(4);
  for (size_t ii = 0; ii != 4; ++ii)
    outline.emplace_back(MapLocation{rot_box.col(ii).x(), rot_box.col(ii).y()});

  map_.convexFillCells(outline, area);

  // now make the area unique
  std::set<int> ros_indices, our_indices;

  for (const auto& cell : area)
    ros_indices.insert(map_.getIndex(cell.x, cell.y));

  for (const auto& cell : cells)
    our_indices.insert(map_.getIndex(cell.x(), cell.y()));

    // compare the size
  ASSERT_EQ(our_indices.size(), ros_indices.size()) << "failed for\n" << rot_box;

  // compare the indices
  for (auto ii = our_indices.begin(), jj = ros_indices.begin();
       ii != our_indices.end(); ++ii, ++jj)
    EXPECT_EQ(*ii, *jj);
}
