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

namespace bresenham {

base_iterator::base_iterator(const cell &_begin, const cell &_end) noexcept {
  const Eigen::Array2i diff = _end - _begin;

  // init the common members. the naming follows the naming of raytrace.
  x_sign = diff.sign().x();
  den = diff.abs().maxCoeff();
  add = diff.abs().minCoeff();

  // some checks
  assert(den >= 0);
  assert(add >= 0);
  assert(den >= add);

  num = den / 2;
  dx = static_cast<double>(diff.x()) / diff.y();
}

x_minor_ascending::x_minor_ascending(const cell &_begin, const cell &_end) :
    base_iterator(_begin, _end) {
  x_curr = _begin.x();
}

x_minor_ascending &
x_minor_ascending::operator++() noexcept {
  num += add;
  if (num >= den) {
    num -= den;
    x_curr += x_sign;
  }

  assert(num < den && "bresenham failed");
  return *this;
}

x_minor_descending::x_minor_descending(const cell &begin, const cell &end) :
    base_iterator(end, begin) {
  x_curr = begin.x();
}

x_minor_descending &
x_minor_descending::operator++() noexcept {
  num -= add;
  if (num < 0) {
    num += den;
    x_curr -= x_sign;
  }

  assert(num > 0 && "bresenham failed");
  return *this;
}

x_major_ascending::x_major_ascending(const cell &_begin, const cell &_end) :
    base_iterator(_begin, _end) {
  x_curr = _begin.x();
}

x_major_ascending &
x_major_ascending::operator++() noexcept {
  const int diff = den - num;
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
  return *this;
}

x_major_descending::x_major_descending(const cell &begin, const cell &end) :
    base_iterator(end, begin) {
  x_curr = begin.x();
}

x_major_descending &
x_major_descending::operator++() noexcept {
  assert(den >= add && "den cannot be smaller then add");

  const int step = num / add;
  num -= (step * add);
  x_curr -= (step * x_sign);
  // check if we need to add another step to reach negative numbers
  if (num >= 0) {
    num -= add;
    x_curr -= x_sign;
  }

  assert(num < 0 && "logic error");
  num += den;
  assert(num > 0 && "logic error");
  return *this;
}

}  // namespace bresenham

x_bresenham::x_bresenham(const cell &_c0, const cell &_c1) noexcept {
  const cell diff = (_c1 - _c0).array().abs();
  if (diff.x() > diff.y()) {
    if (_c1.y() > _c0.y())
      impl_ = std::make_unique<bresenham::x_major_ascending>(_c0, _c1);
    else
      impl_ = std::make_unique<bresenham::x_major_descending>(_c1, _c0);
  }
  else {
    if (_c1.y() > _c0.y())
      impl_ = std::make_unique<bresenham::x_minor_ascending>(_c0, _c1);
    else
      impl_ = std::make_unique<bresenham::x_minor_descending>(_c1, _c0);
  }
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
    if (_l1->get_x() != _l2->get_x())
      return _l1->get_x() < _l2->get_x();

    return _l1->get_dx() < _l2->get_dx();
  }
};

using lines_set = std::list<line *>;

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
    const int l_x = (*l_line)->get_x() + 1;
    const int r_x = (*r_line)->get_x();

    if (l_x >= r_x)
      continue;

    // swipe through the raw costmap.
    const auto r_index = raw_map + _map.getIndex(r_x, yy);
    auto l_index = raw_map + _map.getIndex(l_x, yy);
    for (; l_index != r_index; ++l_index) {
      if (*l_index == _cost)
        return false;
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
check_outline(const costmap_2d::Costmap2D &_map, const polygon &_footprint,
              uint8_t _cost) {
  // get the number of vertices
  const auto cols = _footprint.cols();
  assert(cols > 2 && "no outline defined");

  // setup the check lambda
  auto is_good = [&](const cell __cell) {
    return _map.getCost(__cell.x(), __cell.y()) != _cost;
  };

  for (int ii = 1; ii < cols; ++ii) {
    const auto ray = raytrace(_footprint.col(ii - 1), _footprint.col(ii));
    if (!std::all_of(ray.begin(), ray.end(), is_good))
      return false;
  }

  // the footprint might be closed. in this case we can skip the last vertex,
  // since it does not add any information.
  const auto is_closed = _footprint.col(cols - 1) == _footprint.col(0);
  if (!is_closed) {
    const auto ray = raytrace(_footprint.col(cols - 1), _footprint.col(0));
    return std::all_of(ray.begin(), ray.end(), is_good);
  }
  return true;
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

  // we check the outline seperated, since we have to pay attention to the
  // discretization issues.
  if (!check_outline(_map, _footprint, _cost))
    return false;

  // we are now certain that we have a polygon and will
  // implement the scan-line algorithm below.

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

  // create a map of the vertices mapping the y-value of vertex to its two
  // adjacent edges. assume we have the vertices [a, b, c] and have generated
  // the edges [[a, b], [b, c], [c, a]] which we call [A, B, C]. our map then
  // contains {a.y: [C, A], b.y: [A, B], c.y: [B, C]}.
  using vertex = std::array<line *, 2>;
  std::multimap<int, vertex> vertex_set;
  for (auto l_line = lines.begin(), r_line = std::next(l_line);
       r_line != lines.end(); ++l_line, ++r_line) {
    vertex_set.emplace(r_line->lower.y(), vertex{&(*l_line), &(*r_line)});
  }
  vertex_set.emplace(lines.front().lower.y(),
                     vertex{&lines.back(), &lines.front()});

  // as typical in scan-line algorithm we maintian a set of active lines: the
  // lines are intersected by the current y-value. we sort them by their lowest
  // x-value.
  lines_set active_lines;

  // init the scan-line below the lowerst y-value.
  auto yy = _footprint.row(1).minCoeff() - 1;

  // assume our vertice are [a, b, c]. as above we denote the edges [[a, b], [b,
  // c], [c, a]] as [A, B, C]. the vertices shall have the following coordinates
  // - a: (0, 0)
  // - b: (5, 2)
  // - c: (-1, 3)
  // our yy denotes the current y-value of the scan line and  will range from
  // [-1 ,3]. the vertex_set is  {a.y: [C, A], b.y: [A, B], c.y: [B, C]}. v_next
  // will be the next vertex we can reach.
  // yy = -1:
  //    active_lines: []
  //    v_next: a (with a.y = 0)
  // yy = 0:
  //    active_lines [] -> [A, C]
  //    v_next: b (with b.y = 2)
  // yy = 1:
  //    active_lines [A, C]
  //    v_next: b (with b.y = 2)
  // yy = 2
  //    active_lines [A, C] -> [C, B]
  //    v_next: c (with c.y = 3)
  // yy = 3
  //    active_lines [A, C] -> []
  //    v_next: end

  // iterate over all vertices
  for (auto v_next = vertex_set.begin(); v_next != vertex_set.end();) {
    // our active line set changes only on vertices. we can hence use the
    // current line set until we reach the next vertex.
    for (; yy < v_next->first; ++yy) {
      if (!check_line(_map, active_lines, yy, _cost))
        return false;

      // advance the lines
      for (auto &l : active_lines)
        l->advance_x();
    }

    // we have reached the v_next. we remove ending lines form and add new
    // lines to the active_lines. we iterate until we reach a vertex with a
    // bigger y value, since multiple vertices may be located at the current
    // scan line.
    for (; v_next != vertex_set.end() && v_next->first == yy; ++v_next) {
      // check the two lines of the current vertex
      for (const auto &line : v_next->second) {
        // a line starts, if both if its vertices are equal or above the current
        // scan-line.
        if (line->active) {
          auto iter = std::find(active_lines.begin(), active_lines.end(), line);
          assert(iter != active_lines.end() && "line not found");
          active_lines.erase(iter);
        }
        else {
          line->active = true;
          active_lines.emplace_back(line);
        }
      }
    }
    // reorder the active lines after we have changed them
    active_lines.sort(lines_compare{});
  }
  return true;
}

}  // namespace dpose_core
