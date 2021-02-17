/*
 * MIT License
 *
 * Copyright (c) 2021 Dima Dorezyuk
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include <dpose_core/dpose_costmap.hpp>

#include <cassert>
#include <numeric>
#include <stdexcept>

namespace dpose_core {

polygon
make_footprint(costmap_2d::LayeredCostmap &_cm) {
  const auto res = _cm.getCostmap()->getResolution();
  // resolution cannot be zero otherwise we get device by zero issues
  if (res <= 0)
    throw std::runtime_error("resolution must be positive");

  // convert the message to a eigen-polygon
  const auto fp = _cm.getFootprint();
  polygon cells(2, fp.size());
  for (int cc = 0; cc != cells.cols(); ++cc) {
    cells(0, cc) = std::round(fp.at(cc).x / res);
    cells(1, cc) = std::round(fp.at(cc).y / res);
  }
  return cells;
}

cell_vector
raytrace(const cell &_begin, const cell &_end) noexcept {
  // original code under raytraceLine in
  // https://github.com/ros-planning/navigation/blob/noetic-devel/costmap_2d/include/costmap_2d/costmap_2d.h
  const cell delta_raw = _end - _begin;
  const cell delta = delta_raw.array().abs();

  cell::Index row_major, row_minor;
  const auto den = delta.maxCoeff(&row_major);
  const auto add = delta.minCoeff(&row_minor);

  // this can happen if both are equal. then we can pick whats minor and whats
  // major.
  if (row_major == row_minor)
    row_minor = 1 - row_major;
  const auto size = den;

  assert(size >= 0 && "negative size detected");

  int num = den / 2;

  // setup the increments
  cell inc_minor = delta_raw.array().sign();
  cell inc_major = inc_minor;

  // erase the "other" row
  inc_minor(row_major) = 0;
  inc_major(row_minor) = 0;

  // allocate space beforehand
  cell_vector ray(size);
  cell curr = _begin;

  // bresenham's iteration
  for (int ii = 0; ii < size; ++ii) {
    ray.at(ii) = curr;

    num += add;
    if (num >= den) {
      num -= den;
      curr += inc_minor;
    }
    curr += inc_major;
  }

  return ray;
}

cell_rays
to_rays(const rectangle<int> &_rect) noexcept {
  cell_rays rays;
  for (int cc = 0; cc != 4; ++cc) {
    const auto ray = raytrace(_rect.col(cc), _rect.col(cc + 1));
    for (const auto &cell : ray)
      rays[cell.y()].extend(cell.x());
  }
  return rays;
}

cell_vector
lethal_cells_within(costmap_2d::Costmap2D &_map,
                    const rectangle<int> &_bounds) {
  // enforce that bounds is within costmap
  rectangle<int> search_box = _bounds.array().max(0).matrix();

  // check if size is zero to prevent underflow
  if (_map.getSizeInCellsX() == 0 || _map.getSizeInCellsY() == 0)
    return {};

  // clang-format off
  search_box.row(0) = search_box.row(0).array().min(_map.getSizeInCellsX() - 1).matrix();
  search_box.row(1) = search_box.row(1).array().min(_map.getSizeInCellsY() - 1).matrix();
  // clang-format on

  // first swipe to count the number of elements
  const auto rays = to_rays(search_box);

  size_t count = 0;

  // lock the costmap
  boost::unique_lock<boost::recursive_mutex> lock(*_map.getMutex());
  const auto char_map = _map.getCharMap();
  for (const auto &ray : rays) {
    const auto &y = ray.first;
    // debug-asserts on the indices
    assert(y >= 0 && "y index cannot be negative");
    assert(y < _map.getSizeInCellsY() && "y index out of bounds");
    assert(ray.second.min >= 0 && "x index cannot be negative");
    assert(ray.second.max < _map.getSizeInCellsX() && "x index out of bounds");

    const auto ii_end = _map.getIndex(ray.second.max + 1, y);
    // branchless formulation of "if(char_map[ii] == _value) {++count;}"
    for (auto ii = _map.getIndex(ray.second.min, y); ii != ii_end; ++ii)
      count += (char_map[ii] == costmap_2d::LETHAL_OBSTACLE);
  }

  if (!count)
    return {};

  // resize the final vector
  cell_vector cells(count);

  // dd is the index of the "destination" (where to write to)
  auto dd_iter = cells.begin();

  // write the cells
  for (const auto &ray : rays) {
    const auto &y = ray.first;
    // the conversion to x-value from index is x = index - (y * x_size). the
    // y_offset is here the second part of the equation.
    const auto y_offset = y * _map.getSizeInCellsX();
    const auto ii_end = _map.getIndex(ray.second.max + 1, y);

    // as in the first swipe, but now we convert the index to cells
    for (auto ii = _map.getIndex(ray.second.min, y); ii != ii_end; ++ii) {
      if (char_map[ii] == costmap_2d::LETHAL_OBSTACLE)
        *dd_iter++ = {ii - y_offset, y};
    }
  }

  assert(dd_iter == cells.end() && "bad index: dd_iter");

  return cells;
}

}  // namespace dpose_core