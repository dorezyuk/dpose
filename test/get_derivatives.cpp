#include <laces/laces.hpp>
#include <gtest/gtest.h>

#include <opencv2/imgproc.hpp>

using namespace laces;
using namespace laces::internal;

TEST(get_derivatives, x_line) {
  cv::Mat image(20, 20, cv::DataType<uint8_t>::type, cv::Scalar(0));

  const int y = 10;
  // draw an x line
  cv::line(image, cell_type(0, y), cell_type(image.cols, y), 255);

  const auto edt = euclidean_distance_transform(image);
  const auto d = init_derivatives(edt, cell_type(0, 0));

  // check the result
  // no gradient in x direction
  EXPECT_EQ(cv::countNonZero(d.dx), 0);

  // for y we expect in to decrease in the first half and increase in the second
  for (int cc = 0; cc != image.cols; ++cc) {
    // look at the rows
    for (int rr = 1; rr != y; ++rr)
      EXPECT_LT(d.dy.at<float>(rr, cc), 0) << "bad cell " << cell_type(cc, rr);
    for (int rr = y + 1; rr != image.rows - 1; ++rr)
      EXPECT_GT(d.dy.at<float>(rr, cc), 0) << "bad cell " << cell_type(cc, rr);
  }
}

TEST(get_derivatives, y_line) {
  // this is the counter part the the test above
  cv::Mat image(20, 20, cv::DataType<uint8_t>::type, cv::Scalar(0));

  const int x = 10;
  // draw a line
  cv::line(image, cell_type(x, 0), cell_type(x, image.rows), 255);

  const auto edt = euclidean_distance_transform(image);
  const auto d = init_derivatives(edt, cell_type(0, 0));

  // checks are the same (but flipped rows and cols) as above
  EXPECT_EQ(cv::countNonZero(d.dy), 0);
  for (int rr = 0; rr != image.rows; ++rr) {
    for (int cc = 1; cc != x; ++cc)
      EXPECT_LT(d.dx.at<float>(rr, cc), 0) << "bad cell " << cell_type(cc, rr);
    for (int cc = x + 1; cc != image.cols - 1; ++cc)
      EXPECT_GT(d.dx.at<float>(rr, cc), 0) << "bad cell " << cell_type(cc, rr);
  }
}

TEST(get_derivatives, theta) {
  // here we create a circular edt and expect that most derivatives are close to
  // zero

  cv::Mat image(20, 20, cv::DataType<uint8_t>::type, cv::Scalar(0));
  cell_type center(10, 10);
  image.at<uint8_t>(center) = 255;

  const auto edt = euclidean_distance_transform(image);
  const auto d = init_derivatives(edt, -center);

  EXPECT_EQ(cv::countNonZero(d.dtheta), 0);
}
