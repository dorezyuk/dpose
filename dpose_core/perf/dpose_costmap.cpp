#include <dpose_core/dpose_costmap.hpp>
#include <benchmark/benchmark.h>

using namespace dpose_core;

inline polygon
make_unit_square() {
  polygon square(2, 4);
  square << 1, 1, -1, -1, -1, 1, 1, -1;
  return square;
}

static bool
check_footprint_ros(costmap_2d::Costmap2D& _map,
                    const std::vector<costmap_2d::MapLocation>& _vertices) {
  std::vector<costmap_2d::MapLocation> outline, area;
  _map.polygonOutlineCells(_vertices, outline);
  _map.convexFillCells(outline, area);
  for (const auto& cell : area) {
    if (_map.getCost(cell.x, cell.y) != costmap_2d::LETHAL_OBSTACLE)
      return false;
  }
  return true;
}

// parameterized check for our implementation. the parameter is the scale of a
// unit (2x2) square.
static void
perf_check_footprint_scaled(benchmark::State& state) {
  // get the parameter
  const int scale = state.range(0);

  // setup the costmap
  const costmap_2d::Costmap2D cm(200, 200, 0.1, 0, 0);
  polygon footprint(make_unit_square());

  // scale the footprint
  footprint = (footprint.cast<double>().array() * scale).cast<int>().matrix();

  // place the footprint in the middle
  const Eigen::Vector2i trans(cm.getSizeInCellsX() / 2,
                              cm.getSizeInCellsY() / 2);
  footprint = footprint.colwise() + trans;

  volatile bool is_good = true;
  for (auto _ : state)
    is_good = check_footprint(cm, footprint, costmap_2d::LETHAL_OBSTACLE);
}

// parameterized check for correspondign ros implementation. the parameter is
// the scale of a unit (2x2) square.
static void
perf_check_footprint_scaled_ros(benchmark::State& state) {
  // get the parameter
  const int scale = state.range(0);

  // setup the costmap
  costmap_2d::Costmap2D cm(200, 200, 0.1, 0, 0);
  polygon footprint(make_unit_square());

  // scale the footprint
  footprint = (footprint.cast<double>().array() * scale).cast<int>().matrix();

  // place the footprint in the middle
  const Eigen::Vector2i trans(cm.getSizeInCellsX() / 2,
                              cm.getSizeInCellsY() / 2);
  footprint = footprint.colwise() + trans;

  // now generate the ros-footprint
  std::vector<costmap_2d::MapLocation> vertices;
  vertices.resize(footprint.cols());

  for (size_t ii = 0; ii != vertices.size(); ++ii)
    vertices.at(ii) =
        costmap_2d::MapLocation{footprint(0, ii), footprint(1, ii)};

  volatile bool is_good = true;
  for (auto _ : state)
    is_good = check_footprint_ros(cm, vertices);
}

BENCHMARK(perf_check_footprint_scaled)->DenseRange(1, 50, 5);
BENCHMARK(perf_check_footprint_scaled_ros)->DenseRange(1, 50, 5);

static polygon
make_circle(int _steps, double _radius) noexcept {
  Eigen::Matrix<double, 2, -1> out(2, _steps);

  for (int ii = 0; ii != _steps; ++ii) {
    const auto angle = (M_PI * 2 * ii) / _steps;
    out.col(ii) << std::cos(angle), std::sin(angle);
  }
  // scale
  out *= _radius;
  return out.array().round().cast<int>().matrix();
}

// parametrized check showing the dependency of the runtime to the number of
// points within the original polygon.
static void
perf_check_footprint_dense(benchmark::State& state) {
  // setup the costmap
  const costmap_2d::Costmap2D cm(200, 200, 0.1, 0, 0);

  // create the footprint with the given number of vertices
  const int steps = state.range(0);
  polygon footprint(make_circle(steps, 50));

  // place the footprint in the middle
  const Eigen::Vector2i trans(cm.getSizeInCellsX() / 2,
                              cm.getSizeInCellsY() / 2);
  footprint = footprint.colwise() + trans;

  volatile bool is_good = true;
  for (auto _ : state)
    is_good = check_footprint(cm, footprint, costmap_2d::LETHAL_OBSTACLE);
}

// parametrized check with the corresponding ros implementation showing the
// dependency of the runtime to the number of points within the original
// polygon.
static void
perf_check_footprint_dense_ros(benchmark::State& state) {
  // setup the costmap
  costmap_2d::Costmap2D cm(200, 200, 0.1, 0, 0);

  // create the footprint with the given number of vertices
  const int steps = state.range(0);
  polygon footprint(make_circle(steps, 50));

  // place the footprint in the middle
  const Eigen::Vector2i trans(cm.getSizeInCellsX() / 2,
                              cm.getSizeInCellsY() / 2);
  footprint = footprint.colwise() + trans;

  // now generate the ros-footprint
  std::vector<costmap_2d::MapLocation> vertices;
  vertices.resize(footprint.cols());

  for (size_t ii = 0; ii != vertices.size(); ++ii)
    vertices.at(ii) =
        costmap_2d::MapLocation{footprint(0, ii), footprint(1, ii)};

  volatile bool is_good = true;
  for (auto _ : state)
    is_good = check_footprint_ros(cm, vertices);
}

BENCHMARK(perf_check_footprint_dense)->Arg(10)->Arg(100)->Arg(1000);
BENCHMARK(perf_check_footprint_dense_ros)->Arg(10)->Arg(100)->Arg(1000);
