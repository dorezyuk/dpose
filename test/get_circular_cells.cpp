#include <laces/laces.hpp>
#include <gtest/gtest.h>

using namespace laces;

// file contains tests of the get_circular_cells function

TEST(get_circular_cells, zero) {
  // in the zero case, we want just to return the center cell
  const cell_type center(0, 0);
  const auto cells = get_circular_cells(center, 0);
  ASSERT_FALSE(cells.empty());

  // note: duplicates are ok for now
  for (const auto& cell : cells)
    ASSERT_EQ(cell, center);
}

TEST(get_circular_cells, one) {
  const cell_type center(0, 0);
  const auto cells = get_circular_cells(center, 1);
  ASSERT_FALSE(cells.empty());

  // assemble the expected array
  const cell_vector_type expected = {cell_type{1, 0}, cell_type{0, 1},
                                     cell_type{-1, 0}, cell_type{0, -1}};

  ASSERT_EQ(cells, expected);
}

TEST(get_circular_cells, two) {
  const cell_type center(0, 0);
  const auto cells = get_circular_cells(center, 2);
  ASSERT_FALSE(cells.empty());

  const cell_vector_type expected = {
      cell_type{2, 0},  cell_type{1, 1},   cell_type{0, 2},  cell_type{-1, 1},
      cell_type{-2, 0}, cell_type{-1, -1}, cell_type{0, -2}, cell_type{1, -1}};

  ASSERT_EQ(cells, expected);
}