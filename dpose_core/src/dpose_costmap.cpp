#include <dpose_core/dpose_costmap.hpp>

#include <cassert>
#include <numeric>
#include <stdexcept>

namespace dpose_core {
namespace internal {

polygon
make_footprint(costmap_2d::LayeredCostmap& _cm) {
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
raytrace(const cell& _begin, const cell& _end) noexcept {
  // original code under raytraceLine in
  // https://github.com/ros-planning/navigation/blob/noetic-devel/costmap_2d/include/costmap_2d/costmap_2d.h
  const cell delta_raw = _end - _begin;
  const cell delta = delta_raw.array().abs();

  cell::Index row_major, row_minor;
  const auto den = delta.maxCoeff(&row_major);
  const auto add = delta.minCoeff(&row_minor);
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
to_rays(const rectangle<int>& _rect) noexcept {
  cell_rays rays;
  for (int cc = 0; cc != 4; ++cc) {
    const auto ray = raytrace(_rect.col(cc), _rect.col(cc + 1));
    for (const auto& cell : ray)
      rays[cell.y()].extend(cell.x());
  }
  return rays;
}

}  // namespace internal
}  // namespace dpose_core