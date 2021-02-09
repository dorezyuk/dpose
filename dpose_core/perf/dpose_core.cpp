#include <dpose_core/dpose_core.hpp>
#include <benchmark/benchmark.h>

using namespace dpose_core;
using benchmark::State;

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
  pose_gradient pg(make_arrow(), {3, true});
  pose_gradient::pose se2(0, 0, 0);
  pose_gradient::jacobian J;
  pose_gradient::hessian H;

  // setup the cells
  const cell_vector cells = make_cells();
  // run the tests
  for (auto _ : state)
    pg.get_cost(se2, cells.begin(), cells.end(), &J, &H);
}

static void
perf_dpose_core_no_hessian(State& state) {
  // setup the pg struct
  pose_gradient pg(make_arrow(), {3, true});
  pose_gradient::pose se2(0, 0, 0);
  pose_gradient::jacobian J;

  // setup the cells
  const cell_vector cells = make_cells();

  // run the tests
  for (auto _ : state)
    pg.get_cost(se2, cells.begin(), cells.end(), &J, nullptr);
}

BENCHMARK(perf_dpose_core);
BENCHMARK(perf_dpose_core_no_hessian);

BENCHMARK_MAIN();