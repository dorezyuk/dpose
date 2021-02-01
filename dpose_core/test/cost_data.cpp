#include <dpose_core/dpose_core.hpp>
#include <gtest/gtest.h>

using namespace dpose_core;
using testing::TestWithParam;
using testing::Values;
using testing::Range;

inline polygon
make_triangle() {
  /*
   * (2, 0)
   *   |\
   *   |/  (0, 10)
   * (-2, 0)
   */
  polygon triangle(2, 3);
  triangle << 2, 0, -2, 0, 10, 0;
  return triangle;
}

namespace {
struct translated_triangle : public TestWithParam<cell> {
  cost_data original;
  cost_data translated;
  translated_triangle() {
    // setup the data
    const auto triangle = make_triangle();
    original = cost_data(triangle, 0);
    translated = cost_data(triangle.colwise() + GetParam(), 0);
  }
};

INSTANTIATE_TEST_SUITE_P(/**/, translated_triangle,
                         Values(cell(10, 0), cell(4, 5), cell(-4, -5),
                                cell(0, -9)));

TEST_P(translated_triangle, size_invariance) {
  // verifies that translation of the footprint does not change the size of the
  // image.

  // lazy compare, maybe check the entire values
  EXPECT_EQ(original.get_data().size, translated.get_data().size);
}

TEST_P(translated_triangle, box_variance) {
  // verifies that translation of the footprint is reflected in the get_box
  // return.
  const auto& original_box = original.get_box();
  const auto& translated_box = translated.get_box();

  EXPECT_EQ(original_box.colwise() + GetParam(), translated_box);
}

TEST_P(translated_triangle, origin_variance) {
  // verifies that translation of the footprint is reflected in the get_center
  // return.
  const auto& original_center = original.get_center();
  const auto& translated_center = translated.get_center();

  EXPECT_EQ(original_center - GetParam(), translated_center);
}

}  // namespace

namespace {

struct padded_triangle : public TestWithParam<size_t> {
  cost_data original;
  cost_data padded;

  padded_triangle() {
    // setup the data
    const auto triangle = make_triangle();

    original = cost_data(triangle, 0);
    padded = cost_data(triangle, GetParam());
  }
};

INSTANTIATE_TEST_SUITE_P(/**/, padded_triangle, Values(1, 5, 10));

TEST_P(padded_triangle, size_variance) {
  // verifies that padding is reflected in the final data correctly
  const cv::Size padding(GetParam() * 2, GetParam() * 2);
  const cv::Size original_size(original.get_data().rows,
                               original.get_data().cols);
  const cv::Size padded_size(padded.get_data().rows, padded.get_data().cols);

  // check the image size
  EXPECT_EQ(original_size + padding, padded_size);
}

TEST_P(padded_triangle, box_variance) {
  // verifies that padding is reflected in the get_box call.

  const auto& original_box = original.get_box();
  const auto& padded_box = padded.get_box();

  const auto o_min = original_box.rowwise().minCoeff();
  const auto o_max = original_box.rowwise().maxCoeff();
  const auto p_min = padded_box.rowwise().minCoeff();
  const auto p_max = padded_box.rowwise().maxCoeff();

  // compare the original size with the padded size
  EXPECT_EQ((o_min.array() - GetParam()).matrix(), p_min);
  EXPECT_EQ((o_max.array() + GetParam()).matrix(), p_max);
}

TEST_P(padded_triangle, center_variance) {
  // verifies that padding is reflected in the get_center call.
  const auto& original_center = original.get_center();
  const auto& padded_center = padded.get_center();
  EXPECT_EQ((original_center.array() + GetParam()).matrix(), padded_center);
}

}  // namespace

namespace {

struct rotated_triangle : public TestWithParam<double> {
  cost_data data;
  polygon triangle;
  size_t padding = 5;
  rotated_triangle() {
    // setup the data
    Eigen::Matrix2d rot = Eigen::Rotation2Dd(GetParam()).matrix();
    triangle = (rot * make_triangle().cast<double>()).array().round().matrix().cast<int>();
    data = cost_data(triangle, padding);
  }
};

INSTANTIATE_TEST_SUITE_P(/**/, rotated_triangle, Range(0.1, 3.0, 0.1));

TEST_P(rotated_triangle, triangle) {

  // compute the diffs
  const auto r_min = triangle.rowwise().minCoeff();
  const auto r_max = triangle.rowwise().maxCoeff();
  const auto r_diff = r_max - r_min;
  // get the padded-size
  const auto padded_size = (r_diff.array() + 1 + padding * 2).matrix();

  // compare the size
  EXPECT_EQ(data.get_data().cols, padded_size.x());
  EXPECT_EQ(data.get_data().rows, padded_size.y());

  const auto box_r_min = data.get_box().rowwise().minCoeff();
  const auto box_r_max = data.get_box().rowwise().maxCoeff();

  // compare the box
  EXPECT_EQ((r_min.array() - padding).matrix(), box_r_min);
  EXPECT_EQ((r_max.array() + 1 + padding).matrix(), box_r_max);
}

}  // namespace