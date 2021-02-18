# DposeCore

Library which computes the obstacle cost and the Jacobian of that cost for a polygonal footprint.

## Implementation

The library offers one high-level class: `pose_gradient`.
This class will output the obstacle cost and the Jacobian of that cost for a given polygonal footprint and a vector of obstacle-cells.
The class initialized with a discretized footprint.
The obstacle cost is estimated by a euclidean distance transform from the polygon's outline and stored in a bitmap.
The user should provide some additional padding - in order to penalize also areas close to but not within the footprint.
The cells outside of the footprint are scaled down by 0.01 in order to indicate that that this cost is not-lethal.
This bitmap resides over the lifetime of the class in memory.

## Usage

The basic usage goes like
```cpp
#include <dpose_core/dpose_core.hpp>
using namespace dpose_core;

// setup the parameter struct. set the padding to 5 cells
pose_gradient::parameters param;
param.padding = 5;

// create a footprint: lets use a triangle
polygon triangle(2, 3);
triangle << 2, 0, -2, 0, 10, 0;

// setup the pose-gradient
pose_gradient pg(triangle, param);

// now get the cost for the obstacle costs for a vector of obstacles
cell_vector obstacle_cells{cell{10, 20}, {11, 20}};

// now create the current robot pose. x and y are in cells, z which is theta is
// in rads
pose_gradient::pose robot_pose(5, 5, M_PI);

// now you can get the cost and the jacobian of it
pose_gradient::jacobian j;
const double cost = pg.get_cost(robot_pose, obstacle_cells.cbegin(), obstacle_cells.cend(), j);
```

The output from this call can be now used in non-linear-solver of your choice.
The [dpose_goal_tolerance](../dpose_goal_tolerance) demonstrates the usage with Ipopt.

## Remarks

The initial idea was to pre-compute the first and second partial derivatives on a bitmap (using Sobel operators or similar).
The discretization errors were however, too large for practical usage.
