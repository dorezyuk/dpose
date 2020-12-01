#include <laces/laces.hpp>
#include <gtest/gtest.h>

using namespace laces;

// tests for the draw-polygon function

using testing::TestWithParam;
using testing::Values;

// test where we check that we detect an invalid polygon
struct draw_polygon_invalid : public testing::TestWithParam<size_t> {};

INSTANTIATE_TEST_CASE_P(/**/, draw_polygon_invalid, Values(0, 1, 2));

TEST_P(draw_polygon_invalid, generic) {
  // we expect to throw on an empty
  cell_vector_type cells(GetParam());
  cell_type origin;
  EXPECT_ANY_THROW(draw_polygon(cells, origin));
}

TEST(draw_polygon, simple) {
  // simplest test-case for us
  cell_vector_type cells = {{0, 0}, {10, 5}, {8, 8}};
  cell_type origin;
  const auto image = draw_polygon(cells, origin);

  ASSERT_EQ(image.cols, 11);
  ASSERT_EQ(image.rows, 9);
  ASSERT_EQ(origin, cells.front());

  // check the end-points - they have to be marked
  for (const auto& cell : cells) {
    ASSERT_NE(image.at<uint8_t>(cell), 0) << "bad cell " << cell;
  }

  // repeat the test with a closed polygon - and verify that the output does not
  // change

  cells.push_back(cells.front());
  cell_type origin_closed;
  const auto image_closed = draw_polygon(cells, origin_closed);

  const auto non_equal = image != image_closed;

  ASSERT_EQ(cv::countNonZero(non_equal), 0);
  ASSERT_EQ(origin, origin_closed);
}

TEST(draw_polygon, shift) {
  // as above but we have to shift the origin
  cell_vector_type cells = {{2, 3}, {10, 5}, {8, 8}};
  cell_type origin;
  const auto image = draw_polygon(cells, origin);

  ASSERT_EQ(image.cols, 9);
  ASSERT_EQ(image.rows, 6);
  ASSERT_EQ(origin, cells.front());

  // check the end-points - they have to be marked
  for (const auto& cell : cells) {
    ASSERT_NE(image.at<uint8_t>(cell - cells.front()), 0)
        << "bad cell " << cell;
  }
}

TEST(draw_polygon, negative) {
  // checks if we can do the shifting correclty
  cell_vector_type cells = {{-4, 0}, {10, -5}, {8, 5}};
  cell_type origin;
  const auto image = draw_polygon(cells, origin);

  ASSERT_EQ(image.cols, 15);
  ASSERT_EQ(image.rows, 11);
  ASSERT_EQ(origin, cell_type(-4, -5));
}

int
main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}