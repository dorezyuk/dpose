#include <dpose/dpose.hpp>
#include <gtest/gtest.h>

using namespace dpose;
using namespace dpose::internal;

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

namespace {

// enforce internal binding so we can resuse names

struct draw_polygon_padding_fixture : public TestWithParam<cell_type> {};

INSTANTIATE_TEST_CASE_P(/**/, draw_polygon_padding_fixture,
                        Values(cell_type(1, 2), cell_type(4, 3),
                               cell_type(10, 2)));

TEST_P(draw_polygon_padding_fixture, padding) {
  // here we check the correct handling of the padding
  cell_vector_type cells = {{1, 2}, {10, 5}, {8, 8}};
  const cell_type padding = GetParam();
  cell_type origin;
  const auto image = draw_polygon(cells, origin, padding);

  ASSERT_EQ(image.cols, 10 + padding.x * 2);
  ASSERT_EQ(image.rows, 7 + padding.y * 2);
  ASSERT_EQ(origin, cells.front() - padding);
}

}  // namespace

int
main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}