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

#include <boost/thread/lock_types.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

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

  // setup the increments. note Eigen's signum returns zero if the argument is
  // zero.
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

  // counting first the lethal cells and then writing them into the cell_vector
  // is for the given perf-measurement the fastest way to go.
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

bool
is_inside(const costmap_2d::Costmap2D &_map,
          const polygon &_footprint) noexcept {
  // get the size of the costmap
  const Eigen::Array2i size(_map.getSizeInCellsX(), _map.getSizeInCellsY());

  const int cols = _footprint.cols();
  for (int cc = 0; cc != cols; ++cc) {
    const Eigen::Array2i point = _footprint.col(cc).array();
    if ((point < 0).any() || (point >= size).any())
      return false;
  }
  return true;
}

namespace detail {

x_minor_bresenham::x_minor_bresenham(const cell &c0, const cell &c1) {
  cell diff;

  // we start at the vertex with the lowest y value.
  if (c0.y() < c1.y()) {
    x_curr = c0.x();
    diff = c1 - c0;
  }
  else {
    x_curr = c1.x();
    diff = c0 - c1;
  }

  // init the bresenham's members
  x_sign = diff.array().sign().x();
  den = diff.array().abs().y();
  add = diff.array().abs().x();

  num = den / 2;
  dx = static_cast<double>(diff.x()) / diff.y();
}

int
x_minor_bresenham::get_next() noexcept {
  assert(den >= add && "bad bresenham");

  const auto x_buff = x_curr;
  num += add;
  if (num >= den) {
    num -= den;
    x_curr += x_sign;
  }

  assert(num < den && "bresenham failed");
  return x_buff;
}

x_major_bresenham::x_major_bresenham(const cell &c0, const cell &c1) :
    x_minor_bresenham(c0, c1) {
  // major axis is now x
  std::swap(add, den);
  num = den / 2;
}

int
x_major_bresenham::get_next() noexcept {
  assert(den >= add && "bad bresenham");

  const auto x_buff = x_curr;
  const auto diff = den - num;
  const int step = diff / add;
  num += (step * add);
  x_curr += (step * x_sign);
  // check if we need to add another step to reach den
  if (num < den) {
    num += add;
    x_curr += x_sign;
  }

  assert(num >= den && "logic error");
  num -= den;
  assert(num < den && "logic error");
  return x_buff;
}

}  // namespace detail

x_bresenham::x_bresenham(const cell &_c0, const cell &_c1) noexcept {
  const cell diff = (_c1 - _c0).array().abs();
  if (diff.x() > diff.y())
    impl_ = std::make_unique<detail::x_major_bresenham>(_c0, _c1);
  else
    impl_ = std::make_unique<detail::x_minor_bresenham>(_c0, _c1);
}

struct line : public x_bresenham {
  line(const cell &_begin, const cell &_end) :
      lower(_begin), upper(_end), x_bresenham(_begin, _end) {}
  cell lower, upper;
  bool active = false;
};

struct lines_compare {
  inline bool
  operator()(const line *_l1, const line *_l2) const noexcept {
    if (_l1->get_curr() != _l2->get_curr())
      return _l1->get_curr() < _l2->get_curr();
    return _l1->get_curr() + _l1->get_dx() < _l2->get_curr() + _l2->get_dx();
  }
};

using lines_set = std::set<line *, lines_compare>;

/**
 * @brief helper function for the scan-line algorithm.
 *
 * Function iterates over the _maps cells within a scan-line defined by the
 * index yy. It returns true, if no cell has the value _cost.
 *
 * @param _map The costmap.
 * @param _lines Currently active lines. The size must be even.
 * @param yy The index of the scan-line
 * @param _cost The searched cost.
 */
bool
check_line(const costmap_2d::Costmap2D &_map, const lines_set &_lines, int yy,
           uint8_t _cost) {
  // check the input quality - we need an even size because we swipe between the
  // left and right lines.
  assert(_lines.size() % 2 == 0 && "lines.size() must be even");

  // iterate pair-wise within the active-lines. the active lines are sorted
  // by the x-value.
  auto raw_map = _map.getCharMap();
  for (auto l_line = _lines.begin(); l_line != _lines.end();
       std::advance(l_line, 2)) {
    const auto r_line = std::next(l_line);
    // get the x-values
    const int r_x = (*r_line)->get_next() + 1;
    const int l_x = (*l_line)->get_next();

    // swipe throught the raw costmap.
    const auto r_index = raw_map + _map.getIndex(r_x, yy);
    auto l_index = raw_map + _map.getIndex(l_x, yy);
    for (; l_index != r_index; ++l_index) {
      if (*l_index == _cost)
        *l_index = 100;
      else
        *l_index = 10;
        // return false;
    }
  }
  return true;
}

/**
 * @brief implements a check if a line or point footprint covers a cell with the
 * value _cost.
 *
 * The _footprint must have less then three vertices: it may be empty, a point
 * or a line. The function will check if any cell covered by the footprint has
 * the value _cost. If such cell exists, it returns false.
 *
 * @param _map The costmap.
 * @param _footprint The footprint (defined by vertices in cell coordinates).
 * @param _cost The interesting cost.
 */
bool
check_without_area(const costmap_2d::Costmap2D &_map, const polygon &_footprint,
                   uint8_t _cost) noexcept {
  // get the number of vertices
  const auto cols = _footprint.cols();

  // setup the check lambda
  auto is_good = [&](const cell __cell) {
    return _map.getCost(__cell.x(), __cell.y()) != _cost;
  };

  switch (cols) {
    case 0: {
      // an empty footprint
      return true;
    }
    case 1: {
      // a point-footprint
      return is_good(_footprint.col(0));
    }
    case 2: {
      // a line-footprint
      const auto cells = raytrace(_footprint.col(0), _footprint.col(1));
      return std::all_of(cells.begin(), cells.end(), is_good);
    }
    default: {
      assert(false && "footprint defines an area");
      return false;
    }
  }
}

bool
check_footprint(const costmap_2d::Costmap2D &_map, const polygon &_footprint,
                uint8_t _cost) {
  // check if all vertices are within the costmap
  if (!is_inside(_map, _footprint))
    return false;

  const auto cols = _footprint.cols();
  // if we have less then tree vertices we don't have an area defined.
  if (cols < 3)
    return check_without_area(_map, _footprint, _cost);

  // we are now certain that we have a polygon and will implement the scan-line
  // algorithm below.

  // the footprint might be closed. in this case we can skip the last vertex,
  // since it does not add any information.
  const auto is_closed = _footprint.col(cols - 1) == _footprint.col(0);

  // convert the footprint into lines. a line is defined by two adjacent
  // vertices. assume we have the vertices [a, b, c] (or the closed variant [a,
  // b, c, a]). our edges will be [[a, b], [b, c], [c, a]].
  std::vector<line> lines;
  lines.reserve(cols - is_closed);
  for (int cc = 1; cc != cols; ++cc) {
    // the scan-line algorithm is allergic to horizontal lines; so we skip them.
    if (_footprint(1, cc - 1) != _footprint(1, cc))
      lines.emplace_back(_footprint.col(cc - 1), _footprint.col(cc));
  }
  if (!is_closed && _footprint(1, cols - 1) != _footprint(1, 0))
    lines.emplace_back(_footprint.col(cols - 1), _footprint.col(0));

  // create a height map of the vertices mappint the y-value of vertex to its
  // two adjacent edges. assume we have the vertices [a, b, c] and have
  // generated the edges [[a, b], [b, c], [c, a]] which we call [A, B, C]. our
  // map then contains {a.y: [C, A], b.y: [A, B], c.y: [B, C]}.
  using vertex = std::array<line *, 2>;
  std::multimap<int, vertex> vertex_set;
  // add the first
  vertex_set.emplace(lines.front().lower.y(),
                     vertex{&lines.back(), &lines.front()});
  // add all remaining
  for (auto l_line = lines.begin(), r_line = std::next(l_line);
       r_line != lines.end(); ++l_line, ++r_line)
    vertex_set.emplace(r_line->lower.y(), vertex{&(*l_line), &(*r_line)});

  // as typical in scan-line algorithm we maintian a set of active lines: the
  // lines are intersected by the current y-value. we sort them by their
  // x-value of the lower vertex of the line (from left to right).
  lines_set active_lines;

  // get the y-range of the polygon. since the costmap_2d is row-major, we will
  // iterate within a row in our main loop.
  const auto y_max = _footprint.row(1).maxCoeff() + 1;
  auto next_vertex = vertex_set.begin();
  // assume our vertice are [a, b, c] and have the following coordinates
  // - a: (0, 0)
  // - b: (5, 1)
  // - c: (-1, 2)

  for (auto yy = _footprint.row(1).minCoeff() - 1; yy != y_max;) {
    // our active line set changes only on vertices. we can hence use the
    // current line set until we reach the next vertex.
    for (; yy < next_vertex->first; ++yy)
      if (!check_line(_map, active_lines, yy, _cost))
        return false;
    assert(yy <= y_max && "yy out of range");

    if (next_vertex == vertex_set.end())
      break;
    // we have reached the next_vertex. we remove ending lines form and add new
    // lines to the active set. we iterate until we reach a vertex with a bigger
    // y value, since multiple vertices may be located at the current scan line.
    for (; next_vertex->first == yy; ++next_vertex) {
      for (const auto &line : next_vertex->second) {
        // a line starts, if both if its vertices are equal or above the current
        // scan-line.
        if (line->active) {
          // manual search for the active line as currenlty
          // active_line.find(line) does not work as expected.
          auto iter = active_lines.begin();
          while (iter != active_lines.end() && *iter != line)
            ++iter;
          assert(iter != active_lines.end() && "line not found");
          active_lines.erase(iter);
        }
        else{
          line->active = true;
          active_lines.emplace(line);
        }
      }
    }
  }
  return true;
}

}  // namespace dpose_core
