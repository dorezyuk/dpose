#include <dpose_core/dpose_core.hpp>
#include <dpose_core/dpose_costmap.hpp>
#include <benchmark/benchmark.h>

#include <costmap_2d/cost_values.h>
#include <costmap_2d/costmap_2d.h>

using namespace dpose_core;

using benchmark::State;
using costmap_2d::Costmap2D;
using costmap_2d::LETHAL_OBSTACLE;
using costmap_2d::MapLocation;

namespace {

inline polygon
make_arrow() {
  /*
   *  (-60, 20)    (40, 20)
   * (-30, 0) \--\
   *          /--/ (50, 0)
   * (-60, -20)    (40, -20)
   */
  polygon arrow(2, 6);
  // clang-format off
    arrow << 50,  40, -60, -30, -60, 40,
             0, -20, -20,  0,  20, 20;
  // clang-format on
  return arrow;
}

cell_vector
make_cells() {
  cell_vector cells;
  cells.reserve(100);
  for (int xx = 0; xx != 10; ++xx)
    for (int yy = 0; yy != 10; ++yy)
      cells.emplace_back(xx, yy);
  return cells;
}

}  // namespace

static void
perf_dpose_core(State& state) {
  // setup the pg struct
  pose_gradient pg(make_arrow(), {3});
  pose_gradient::pose se2(0, 0, 0);
  pose_gradient::jacobian J;

  // setup the cells
  const cell_vector cells = make_cells();
  // run the tests
  for (auto _ : state)
    pg.get_cost(se2, cells.begin(), cells.end(), &J);
}

static void
perf_dpose_core_no_jacobian(State& state) {
  // setup the pg struct
  pose_gradient pg(make_arrow(), {3});
  pose_gradient::pose se2(0, 0, 0);

  // setup the cells
  const cell_vector cells = make_cells();

  // run the tests
  for (auto _ : state)
    pg.get_cost(se2, cells.begin(), cells.end(), nullptr);
}

static void
perf_dpose_costmap(State& state) {
  // setup the costmap
  Costmap2D map(200, 200, 0.1, 0, 0);

  // mark every second cell as lethal
  const auto size = map.getSizeInCellsX() * map.getSizeInCellsY();
  auto m = map.getCharMap();

  for (size_t ii = 0; ii < size; ii += 2)
    m[ii] = LETHAL_OBSTACLE;

  // query the cells
  rectangle<int> box = to_rectangle(198, 198);

  for (auto _ : state)
    volatile auto cells = lethal_cells_within(map, box);
}

// just for refernce: the ros-impl which just returns the cells
static void
perf_dpose_costmap_ros(State& state) {
  // setup the costmap
  Costmap2D map(200, 200, 0.1, 0, 0);

  std::vector<MapLocation> outline, area;
  rectangle<int> box = to_rectangle(198, 198);

  // setup the outline
  outline.reserve(4);
  for (size_t ii = 0; ii != 4; ++ii)
    outline.emplace_back(MapLocation{box.col(ii).x(), box.col(ii).y()});

  for (auto _ : state){
    map.convexFillCells(outline, area);
    area.clear();
  }
}

BENCHMARK(perf_dpose_core);
BENCHMARK(perf_dpose_core_no_jacobian);
BENCHMARK(perf_dpose_costmap);
BENCHMARK(perf_dpose_costmap_ros);

BENCHMARK_MAIN();