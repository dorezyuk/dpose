#include <laces/laces.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace laces {

cv::Mat
draw_polygon(const cell_vector_type& _cells, cell_type& _shift) {
  // we need an area - so at least three points
  if (_cells.size() < 3)
    throw std::invalid_argument("cells must define a valid area");

  cv::Rect2d bb = cv::boundingRect(_cells);

  // we cannot do much if the input does not result in a valid bounding box
  if (bb.empty())
    throw std::invalid_argument("bb cannot be empty");

  // setup the image
  cv::Scalar s(0);
  cv::Mat image(bb.size(), cv::DataType<uint8_t>::type, s);

  // it may be that the _cells contain negative numbers - we cannot draw them
  // so we have to shift everything by the lower left corner of the bounding box
  _shift = static_cast<cell_type>(bb.tl());
  auto shifted_cells = _cells;
  for (auto& cell : shifted_cells)
    cell -= _shift;

  // draw the polygon into the image
  cv::polylines(image, shifted_cells, true, 255);

  return image;
}

inline bool
is_valid(const cell_type& _cell, const cv::Mat& _image) noexcept {
  return 0 <= _cell.x && _cell.x < _image.cols && 0 <= _cell.y &&
         _cell.y < _image.rows;
}

/// @brief helper which stores the current location and the origin
struct voronoi_cell {
  cell_type self;
  cell_type origin;
};

cv::Mat
euclidean_distance_transform(cv::InputArray _image) {
  cv::Mat seen;
  cv::Mat edt(_image.size(), cv::DataType<float>::type,
              cv::Scalar(std::numeric_limits<float>::max()));
  // note: implementation is not very fast, but thats also not where the speed
  // is at for this library.

  // init the occupied cells
  cell_vector_type seen_cells;
  cv::threshold(_image, seen, 254, 255, cv::THRESH_BINARY);
  cv::findNonZero(seen, seen_cells);

  // init the edt image
  edt.setTo(0, seen);

  // reset seen
  seen.setTo(0);

  // setup the neighbors on a eight-connected grid
  const std::array<cell_type, 8> neighbors = {
      cell_type{-1, 0}, cell_type{1, 0},  cell_type{0, -1}, cell_type{0, 1},
      cell_type{1, 1},  cell_type{-1, 1}, cell_type{1, -1}, cell_type{-1, -1}};

  // todo make sure that this is a shallow copy
  const auto mat = _image.getMat();

  // init the curr wave
  std::vector<voronoi_cell> curr(seen_cells.size()), next;
  std::transform(seen_cells.begin(), seen_cells.end(), curr.begin(),
                 [](const cell_type& __cell) {
                   return voronoi_cell{__cell, __cell};
                 });

  // do a voronoi-style expansion
  while (!curr.empty()) {
    // iterate over all cells within the current wave
    for (const auto& cell : curr) {
      // skip if this cells is a duplicate
      if (seen.at<uint8_t>(cell.self) &&
          edt.at<float>(cell.self) < cv::norm(cell.self - cell.origin))
        continue;

      // mark this cells as seen
      seen.at<uint8_t>(cell.self) = 1;

      // iterate over all neighbors
      for (const auto& offset : neighbors) {
        const auto candidate = cell.self + offset;

        // skip of of bound-candidates
        if (!is_valid(candidate, mat))
          continue;

        const auto dist =
            static_cast<float>(cv::norm(cell.self + offset - cell.origin));
        // skip cells if we dont improve
        if (edt.at<float>(candidate) <= dist)
          continue;

        // mark the current cell
        edt.at<float>(candidate) = dist;
        next.emplace_back(voronoi_cell{candidate, cell.origin});
      }
    }
    // move will invalidate next...
    curr = std::move(next);
  }

  return edt;
}

derivatives
get_derivatives(cv::InputArray _image) {
  derivatives d;
  cv::Sobel(_image, d.dx, cv::DataType<float>::type, 1, 0, 3, 10);
  cv::Sobel(_image, d.dy, cv::DataType<float>::type, 0, 1, 3, 10);
  return d;
}

}  // namespace laces