#include <laces/laces.hpp>
#include <gtest/gtest.h>

using namespace laces;

TEST(get_derivatives, generic) {
  cell_vector_type cells = {{0, 0}, {50, 50}, {30, 70}};
  cell_type origin;
  const auto im1 = draw_polygon(cells, origin);
  const auto im2 = euclidean_distance_transform(im1);
  const auto d = get_derivatives(im2);

  cv::imwrite("dx.png", d.dx + cv::Scalar(100));
  cv::imwrite("dy.png", d.dy);

}