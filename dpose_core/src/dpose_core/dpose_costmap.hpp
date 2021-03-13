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
#ifndef DPOSE_CORE__DPOSE_COSTMAP__HPP
#define DPOSE_CORE__DPOSE_COSTMAP__HPP

#include <dpose_core/dpose_core.hpp>

#include <costmap_2d/layered_costmap.h>

#include <cmath>
#include <limits>
#include <unordered_map>

namespace dpose_core {

/// @brief generates a polygon from a costmap_2d::LayeredCostmap.
/// @param _cm costmap - which we *don't* alter.
/// @throws std::runtime_error if the resolution is negative or zero.
polygon
make_footprint(costmap_2d::LayeredCostmap& _cm);

/// @brief bresenham's raytrace algorithm adjusted from ros.
/// @param _begin begin of the ray.
/// @param _end end of the ray (exclusive).
cell_vector
raytrace(const cell& _begin, const cell& _end) noexcept;

/**
 * @brief interval defined by [min, max].
 *
 * Use interval::extend in order to add values.
 * We will use this structure in order to iterate on a ray defined by
 * its y-value and x_begin and x_end.
 */
template <typename _T>
struct interval {
  interval() :
      min{std::numeric_limits<_T>::max()},
      max{std::numeric_limits<_T>::lowest()} {}

  inline void
  extend(const _T& _v) noexcept {
    min = std::min(min, _v);
    max = std::max(max, _v);
  }

  _T min, max;
};

using cell_interval = interval<size_t>;
using cell_rays = std::unordered_map<int, cell_interval>;
using cell_rectangle = rectangle<int>;

/// @brief constructs intervals for every y value from _rect
/// @param _rect the rectangle
cell_rays
to_rays(const cell_rectangle& _rect) noexcept;

/// @brief returns a vector with lethal cells within the box
/// @param _map the costmap
/// @param _bounds box in cell-space defining the ROI
cell_vector
lethal_cells_within(costmap_2d::Costmap2D& _map, const cell_rectangle& _bounds);

/// @brief checks if the _footprint is entirely inside the _map.
/// @param _map the costmap.
/// @param _footprint the footprint (in cell-space).
bool
is_inside(const costmap_2d::Costmap2D& _map,
          const polygon& _footprint) noexcept;

namespace bresenham {

/// @brief base class for our customized bresenham iteratorion.
struct base_iterator {
  base_iterator(const cell& _begin, const cell& _end) noexcept;
  virtual ~base_iterator() = default;

  inline const int&
  operator*() const noexcept {
    return x_curr;
  }

  inline const double&
  get_dx() const noexcept {
    return dx;
  }

  virtual base_iterator&
  operator++() noexcept = 0;

protected:
  int x_curr, x_sign, den, add, num;
  double dx;
};

struct x_minor_ascending : public base_iterator {
  x_minor_ascending(const cell& _begin, const cell& _end);

  x_minor_ascending&
  operator++() noexcept final;
};

struct x_minor_descending : public base_iterator {
  x_minor_descending(const cell& _begin, const cell& _end);

  x_minor_descending&
  operator++() noexcept final;
};

struct x_major_ascending : public base_iterator {
  x_major_ascending(const cell& _begin, const cell& _end);

  x_major_ascending&
  operator++() noexcept final;
};

struct x_major_descending : public base_iterator {
  x_major_descending(const cell& begin, const cell& end);

  x_major_descending&
  operator++() noexcept final;
};

}  // namespace bresenham

/**
 * @brief Implements the bresenham-raytracing algorithm such that the get_next()
 * method will always yield the next x-value on a line.
 *
 * The sequence will start with the x-value of the given vertex which has the
 * lowerst y-value.
 */
struct x_bresenham {
  x_bresenham(const cell& _c0, const cell& _c1) noexcept;

  inline void
  advance_x() noexcept {
    impl_->operator++();
  }

  inline const int&
  get_x() const noexcept {
    return **impl_;
  }

  inline const double&
  get_dx() const noexcept {
    return impl_->get_dx();
  }

private:
  std::unique_ptr<bresenham::base_iterator> impl_;
};

bool
check_footprint(const costmap_2d::Costmap2D& _map, const polygon& _footprint,
                uint8_t _cost);

}  // namespace dpose_core

#endif  // DPOSE_CORE__DPOSE_COSTMAP__HPP
