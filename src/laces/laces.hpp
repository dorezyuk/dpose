#pragma once

// we will use boost::geometry for representing and manipulating the footprint
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometry.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

namespace laces {

// define the boost types
namespace bg = boost::geometry;

using point_type = bg::model::d2::point_xy<double>;
using polygon_type = bg::model::polygon<point_type>;

using cell_type = cv::Point2i;
using cell_vector_type = std::vector<cell_type>;

/**
 * @brief Returns an image with a polygon defined by _cells
 *
 * The image will have the size of the bounding box of _cells.
 * The drawn polygon will be closed and the value of the line is 255.
 * The function will shift the origin such that the smallest cell in _cells is
 * [0, 0]. The origin is then moved by new_origin = old_origin - _shift.
 *
 * @param[in] _cells an (open) polygon
 * @param[out] _shift shifted origin of the _cells
 *
 * @return cv::Mat image showing the polygon
 * @throw std::invalid_argument if _cells is empty
 */
cv::Mat
draw_polygon(const cell_vector_type& _cells, cell_type& _shift);

/**
 * @brief Returns a edt from the input image.
 *
 * The function will only apply the edt to pixels with the value 255.
 * Use draw_polygon to generate a valid input.
 *
 * @param _image input image
 * @return cv::Mat edt generated from _image
 */
cv::Mat
euclidean_distance_transform(cv::InputArray _image);

cell_vector_type
get_circular_cells(const cell_type& _center, int _radius);

cv::Mat
angular_derivative(cv::InputArray _image, const cell_type& _center);

/**
 * @brief POD holding the derivatives
 */
struct derivatives {
  cv::Mat dx;        ///< derivative in x
  cv::Mat dy;        ///< derivative in y
  cv::Mat dtheta;    ///< derivative in theta
  cell_type center;  ///< center of rotation
};

derivatives
init_derivatives(cv::InputArray _image, const cell_type& _center);

/**
 * @brief Highest api level for the user to the get the derivatives
 *
 * @param _cells
 * @return derivatives
 */
derivatives
init_derivatives(const cell_vector_type& _cells);

using cost_type = cv::Scalar_<float>;

/**
 */
cost_type
get_derivative(const derivatives& _data, const cell_vector_type& _cells);

float
get_cost(const cv::Mat& _data, const cell_vector_type& _cells);

}  // namespace laces