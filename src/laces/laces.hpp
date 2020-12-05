#pragma once

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

namespace laces {

using cell_type = cv::Point2i;
using cell_vector_type = std::vector<cell_type>;

/// @brief POD holding the derivatives
struct derivatives {
  cv::Mat dx;        ///< derivative in x
  cv::Mat dy;        ///< derivative in y
  cv::Mat dtheta;    ///< derivative in theta
  cell_type center;  ///< center of rotation
};

namespace internal {
// code inside this namespace is not part of the public interface.
// we put it here so we can actually unit-test it.

/**
 * @brief Returns an image with a polygon defined by _cells
 *
 * The image will have the size of the bounding box of _cells.
 * The drawn polygon will be closed and the value of the line is 255.
 * The function will shift the origin such that the smallest cell in _cells is
 * [0, 0]. The origin is then moved by new_origin = old_origin - _shift.
 *
 * @param[in] _cells an polygon
 * @param[out] _shift shifted origin of the _cells
 * @param[in] _padding optional padding to each side. padding cannot be
 * negative.
 *
 * @return cv::Mat image showing the polygon
 * @throw std::invalid_argument if _cells is empty or if padding is negative
 */
cv::Mat
draw_polygon(const cell_vector_type& _cells, cell_type& _shift,
             const cell_type& _padding = cell_type(0, 0));

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

cv::Mat
smoothen_edges(cv::InputArray _image, const cell_vector_type& _cells);

/**
 * @brief Returns cells laying on the perimeter of a circle.
 *
 * Function implements the circular bresenham algorithm. It returns all cells
 * at a circle defined by the _center and _radius. Those cells are sorted CW,
 * and without duplicates.
 *
 * @param _center center of the circle
 * @param _radius radius of the circle
 * @return cells at the perimeter of the circle
 */
cell_vector_type
get_circular_cells(const cell_type& _center, size_t _radius) noexcept;

/**
 * @brief Computes the circular derivative around the _center.
 *
 * The derivative for a cell is approximated by the difference if its neighbor
 * cells, laying in the same circle (around the _center cell).
 *
 * @param _image input for the derivative calculation
 * @param _center center of rotation
 * @return matrix with the angular derivate of _image around _center.
 */
cv::Mat
angular_derivative(cv::InputArray _image, const cell_type& _center);

/**
 * @brief Helper function to create the derivatives from an image.
 *
 * @param _image image for the derivative calculation
 * @param _center center of rotation
 * @return derivatives
 */
derivatives
init_derivatives(cv::InputArray _image, const cell_type& _center);

}  // namespace internal

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
 * @brief Get the derivative object
 *
 * @param _data
 * @param _cells
 * @return cost_type
 */
cost_type
get_derivative(const derivatives& _data, const cell_vector_type& _cells);

/**
 * @brief Get the cost object
 *
 * @param _data
 * @param _cells
 * @return float
 */
float
get_cost(const cv::Mat& _data, const cell_vector_type& _cells);

}  // namespace laces