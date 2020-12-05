#include <laces/laces.hpp>
#include <gtest/gtest.h>

using namespace laces;
using namespace laces::internal;

TEST(euclidean_distance_transform, simple) {
  // setup the image with a polygon
  cell_vector_type cells = {{0, 0}, {500, 500}, {300, 700}};
  cell_type origin;
  const auto im1 = draw_polygon(cells, origin);
  const auto edt = euclidean_distance_transform(im1);
  double max_v, min_v;
  cv::minMaxLoc(edt, &min_v, &max_v);

  // we expect that no value is infinite
  ASSERT_EQ(min_v, 0);

  // max value cannot be zero but must be smaller then the overall image
  ASSERT_LE(max_v, std::hypot(edt.rows, edt.cols));
  ASSERT_GT(max_v, 0);

  // check the cells (very lazy check...)
  for (const auto& cell : cells) {
    ASSERT_EQ(edt.at<float>(cell), 0) << "bad cell " << cell;
  }
}

using testing::TestWithParam;
using testing::Values;

// this is the fixture to perform (slow) regression tests
struct euclidean_distance_transform_fixture
    : public TestWithParam<cell_vector_type> {};

INSTANTIATE_TEST_CASE_P(
    /**/, euclidean_distance_transform_fixture,
    Values(
        cell_vector_type{{10, 10}},                      // single point
        cell_vector_type{{10, 10}, {11, 11}, {12, 12}},  // diagonal line
        cell_vector_type{{10, 10}, {11, 9}, {12, 8}},  // reversed diagonal line
        cell_vector_type{{10, 10}, {12, 10}, {12, 12}, {10, 12}},  // box
        cell_vector_type{{5, 5}, {15, 12}, {13, 7}, {9, 8}}        // points
        ));

TEST_P(euclidean_distance_transform_fixture, regression) {
  // setup the matrix
  cv::Mat image(20, 25, cv::DataType<uint8_t>::type, cv::Scalar(0));

  // mark the cells
  const auto cells = GetParam();
  for (const auto& cell : cells)
    image.at<uint8_t>(cell) = 255;

  // perform the edt operation
  const auto edt = euclidean_distance_transform(image);

  // check the result
  for (int rr = 0; rr != image.rows; ++rr)
    for (int cc = 0; cc != image.cols; ++cc) {
      cell_type curr(cc, rr);
      // find the expected value
      auto dist = std::numeric_limits<float>::max();

      for (const auto cell : cells)
        dist = std::min<float>(dist, cv::norm(curr - cell));

      EXPECT_FLOAT_EQ(dist, edt.at<float>(curr)) << "bad cell " << curr;
    }
}