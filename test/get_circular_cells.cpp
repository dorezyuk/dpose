#include <laces/laces.hpp>
#include <gtest/gtest.h>

using namespace laces;
using namespace laces::internal;

// file contains tests of the get_circular_cells function

// force internal binding, so we dont get conflicting names
namespace {

// parameter POD with input and expected output
struct parameter {
  int radius;
  cell_vector_type expected;
};

// setup the regression input and results
parameter params[] = {
    parameter{0, {cell_type{0, 0}}},
    parameter{
        1,
        {cell_type{1, 0}, cell_type{0, 1}, cell_type{-1, 0}, cell_type{0, -1}}},
    parameter{2,
              {cell_type{2, 0}, cell_type{1, 1}, cell_type{0, 2},
               cell_type{-1, 1}, cell_type{-2, 0}, cell_type{-1, -1},
               cell_type{0, -2}, cell_type{1, -1}}},
    parameter{
        3,
        {cell_type{3, 0}, cell_type{2, 1}, cell_type{2, 2}, cell_type{1, 2},
         cell_type{0, 3}, cell_type{-1, 2}, cell_type{-2, 2}, cell_type{-2, 1},
         cell_type{-3, 0}, cell_type{-2, -1}, cell_type{-2, -2},
         cell_type{-1, -2}, cell_type{0, -3}, cell_type{1, -2},
         cell_type{2, -2}, cell_type{2, -1}}}};

using testing::TestWithParam;
using testing::ValuesIn;

// simple test-fixture
struct get_circular_cells_fixture : public TestWithParam<parameter> {};

INSTANTIATE_TEST_CASE_P(/**/, get_circular_cells_fixture, ValuesIn(params));

TEST_P(get_circular_cells_fixture, regression) {
  // get the params
  const auto param = GetParam();
  const cell_type center(0, 0);

  // call the function
  const auto cells = get_circular_cells(center, param.radius);

  // check the result
  ASSERT_EQ(cells, param.expected);
}

}  // namespace
