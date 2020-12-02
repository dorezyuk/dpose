#include <laces/laces.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
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

/**
 * @brief Retuns the maximum euclidean distance from the cell to the image
 * corners
 *
 * @param _image the image
 * @param _cell the cell
 * @return maximum distance from the cell to the image corners
 */
double
max_distance(cv::InputArray _image, const cell_type& _cell) {
  const auto m = _image.getMat();
  // get the corners
  std::array<cell_type, 4> corners{cell_type{0, 0}, cell_type{0, m.rows},
                                   cell_type{m.cols, 0},
                                   cell_type{m.cols, m.rows}};

  // get the closest distance
  auto dist = 0.;
  for (const auto& corner : corners)
    dist = std::max(cv::norm(corner - _cell), dist);

  return dist;
}

void
unique_cells(cell_vector_type& _cells) {
  if (_cells.empty())
    return;
  // use first unique to remove the overlaps
  auto last = std::unique(_cells.begin(), _cells.end());

  // now check if we loop - see if the first cells reappears
  last = std::find(std::next(_cells.begin()), last, _cells.front());

  // drop the redundant data
  _cells.erase(last, _cells.end());
}

cell_vector_type
get_circular_cells(const cell_type& _center, int _radius) {
  // adjusted from
  // https://github.com/opencv/opencv/blob/master/modules/imgproc/src/drawing.cpp
  int x = _radius, y = 0;
  int err = 0, plus = 1, minus = (_radius << 1) - 1;

  std::array<cell_vector_type, 8> octets;

  // the order is
  // [[x,y], [y, x], [y, -x], [-x, y], [-x, -y], [-y, -x], [-y, x],  [x, -y]]
  while (x >= y) {
    // insert the octets - for now without fancy looping
    auto octet = octets.begin();
    // clang-format off
    octet->emplace_back(_center.x + x, _center.y + y); ++octet;
    octet->emplace_back(_center.x + y, _center.y + x); ++octet;
    octet->emplace_back(_center.x - y, _center.y + x); ++octet;
    octet->emplace_back(_center.x - x, _center.y + y); ++octet;

    octet->emplace_back(_center.x - x, _center.y - y); ++octet;
    octet->emplace_back(_center.x - y, _center.y - x); ++octet;

    octet->emplace_back(_center.x + y, _center.y - x); ++octet;
    octet->emplace_back(_center.x + x, _center.y - y); ++octet;
    // clang-format on

    ++y;
    err += plus;
    plus += 2;

    int mask = (err <= 0) - 1;

    err -= minus & mask;
    x += mask;
    minus -= mask & 2;
  }

  // now flatten the octets
  cell_vector_type cells;
  cells.reserve(octets.begin()->size() * octets.size());
  // we have to reverse every second octet
  bool reverse = false;
  for (const auto& octet : octets) {
    if (reverse)
      cells.insert(cells.end(), octet.rbegin(), octet.rend());
    else
      cells.insert(cells.end(), octet.begin(), octet.end());
    reverse = !reverse;
  }

  unique_cells(cells);
  return cells;
}

cv::Mat
angular_derivative(cv::InputArray _image, const cell_type& _center) {
  // todo - this is not the end, but we are too lazy to deal with it now.
  // we would have to shift the image later...
  if (_center.x > 0 || _center.y > 0)
    throw std::invalid_argument("invalid center cell");

  const cell_type center = -_center;

  // get the distance
  // todo this must never get negative...
  const auto distance = static_cast<size_t>(max_distance(_image, center));
  return cv::Mat();
}

/**
 * @brief helper function to get the derivative from an image
 */
derivatives
init_derivatives(cv::InputArray _image, const cell_type& _center) {
  derivatives d;

  // x and y derivatives are really easy...
  cv::Sobel(_image, d.dx, cv::DataType<float>::type, 1, 0, 3, 10);
  cv::Sobel(_image, d.dy, cv::DataType<float>::type, 0, 1, 3, 10);
  d.center = _center;
  // d.dtheta = angular_derivative(_image, _center);
  return d;
}

derivatives
init_derivatives(const cell_vector_type& _cells) {
  cell_type center;
  const auto im1 = draw_polygon(_cells, center);
  const auto im2 = euclidean_distance_transform(im1);
  return init_derivatives(im2, center);
}

/**
 * @brief todo document me
 */
inline cost_type
get_derivative(const derivatives& _data, const cell_type& _cell) {
  if (!is_valid(_cell, _data.dx))
    return {0, 0, 0};

  return {_data.dx.at<float>(_cell), _data.dy.at<float>(_cell),
          _data.dtheta.at<float>(_cell)};
}

cost_type
get_derivative(const derivatives& _data, const cell_vector_type& _cells) {
  cost_type out(0, 0, 0);

  for (const auto& cell : _cells)
    out += get_derivative(_data, cell);

  return out;
}

/**
 * @brief todo document me
 */
inline float
get_cost(const cv::Mat& _data, const cell_type& _cell) {
  // invalid cells are ok for us (the image might be rotated...)
  if (!is_valid(_cell, _data))
    return 0;
  return _data.at<float>(_cell);
}

float
get_cost(const cv::Mat& _data, const cell_vector_type& _cells) {
  // accumulate the cost over the entire cell vector
  return std::accumulate(_cells.begin(), _cells.end(), 0.f,
                         [&](float _sum, const cell_type& _cell) {
                           return _sum + get_cost(_data, _cell);
                         });
}

}  // namespace laces