// Jetson smoke test: does the actual sweep footprint build, solve, and fit in memory.
//
// NOT a benchmark -- no comparable numbers are produced, and none should be read from this file.
// It exists to answer three yes/no questions per planner, on whatever config.h was written before
// the build (scripts/run_jetson_smoke_test.sh writes the sweep's real MODEL 1 / MAX_TREE_SIZE
// 3000000 / large-delta config, so this measures the footprint that matters):
//
//   1. DID IT SOLVE. Not "no CUDA error" -- a planner that runs 50 clean iterations without ever
//      reaching the goal is not a passing smoke test, and the empty environment with a config-
//      derived start/goal (see below) makes "never reaches the goal" a real failure signal rather
//      than a config mismatch.
//   2. DID IT LEAK. Planner has no destructor of its own historically, so d_randomSeeds_ptr_ alone
//      (48B * MAX_TREE_SIZE) leaked on every construction -- 144 MiB at this config, times however
//      many planners a sweep constructs. cudaMemGetInfo before/after each planner's scope is the
//      direct check, and it is what makes the ~Planner() fix in Planner.cu verifiable rather than
//      assumed.
//   3. DID IT HANG. A wall-clock budget per planner, matching every real benchmark harness in this
//      repo (countingstars_sweep.cu, kinopaxplus_delta_benchmark.cu all carry one; this test alone
//      did not).
//
// FOUR DIFFERENT LOOP SHAPES, not one generic template plus a special case, because the planners
// genuinely differ in two independent ways and pretending otherwise would silently mis-test some of
// them:
//
//   - solve signal:      d_pathToGoal_/h_pathToGoal_ (KPAX, PruneKPAX)
//                    vs  d_minCost_ptr_/h_minCost_   (KinoPaxSTAR family, KinoPaxPlus, CountingStars)
//   - graph_.updateVertices(): called by every Graph-based planner (KPAX, PruneKPAX, KinoPaxSTAR,
//                    KinoPaxSTARCleanCost) EXCEPT KinoPaxPlus (different graph type, no vertexScores)
//                    and CountingStars (Graph-based but deliberately never calls it -- see
//                    countingstars_sweep.cu's benchmarkCountingStars for why: it consumes nothing
//                    updateVertices() produces, and the kernel is not cheap).
//
// Each loop body below is copied from the real, tested pattern in examples/gpu/countingstars_sweep.cu
// (benchmarkKPAX / benchmarkKinoPaxPlus / benchmarkKinoPaxSTARCleanCost / benchmarkCountingStars),
// not reinvented, so a planner that would fail there fails identically here.
#include <iostream>
#include <string>
#include <chrono>
#include "planners/KPAX.cuh"
#include "planners/PruneKPAX.cuh"
#include "planners/KinoPaxPlus.cuh"
#include "planners/KinoPaxSTAR.cuh"
#include "planners/KinoPaxSTARCleanCost.cuh"
#include "planners/CountingStars.cuh"

// Per-planner wall-clock budget. Matches the class of timeout every real benchmark harness in this
// repo carries (countingstars_sweep.cu's MAX_TIME_MS, kinopaxplus_delta_benchmark.cu's); this test
// previously had none, so a hang here would have blocked forever rather than failing.
static const double MAX_WALL_SECONDS = 15.0;

// Tolerance on the before/after cudaMemGetInfo comparison. Not zero: the CUDA driver's allocator
// can retain a small pool between a cudaFree and the next cudaMemGetInfo, and that drift is
// unrelated to the bug this test exists to catch. 64 MiB is generous headroom above that drift and
// still well under half of the ~144 MiB a single un-freed d_randomSeeds_ptr_ leaks at this config
// -- so a real leak fails loudly and allocator noise does not fail spuriously.
static const size_t LEAK_TOLERANCE_BYTES = 64ULL * 1024 * 1024;

// Print free/total GPU memory in MB
void printGPUMemory(const char* label)
{
    size_t freeMem = 0, totalMem = 0;
    cudaMemGetInfo(&freeMem, &totalMem);
    printf("  [Memory %s] Free: %.1f MB / Total: %.1f MB\n",
           label, freeMem / (1024.0 * 1024.0), totalMem / (1024.0 * 1024.0));
}

bool checkCudaError(const char* plannerName)
{
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess)
        {
            printf("  FAIL - %s: CUDA error: %s\n", plannerName, cudaGetErrorString(err));
            return false;
        }
    return true;
}

// Shared by every loop shape below: prints memory before, runs `body` (which returns true once the
// planner reports a solution), enforces the wall-clock budget, and checks for a leak after the
// planner's scope has closed. `body` owns constructing/destroying the planner so its destructor
// runs before the "after" memory reading.
template<typename BodyFn>
bool runSmokeTestFramed(const char* name, BodyFn body)
{
    printf("\n--- %s ---\n", name);
    printGPUMemory("before");
    size_t freeBefore = 0, totalMem = 0;
    cudaMemGetInfo(&freeBefore, &totalMem);

    bool solved = false;
    bool timedOut = false;
    int itr = 0;

    auto wallStart = std::chrono::steady_clock::now();
    body(solved, timedOut, itr, wallStart);

    if(!checkCudaError(name)) return false;

    cudaDeviceSynchronize();
    printGPUMemory("after");

    if(!checkCudaError(name)) return false;

    size_t freeAfter = 0;
    cudaMemGetInfo(&freeAfter, &totalMem);
    // freeAfter should be >= freeBefore minus tolerance. freeBefore > freeAfter means memory is
    // still held after the planner's destructor ran -- a leak.
    bool leaked = (freeBefore > freeAfter) && (freeBefore - freeAfter > LEAK_TOLERANCE_BYTES);

    if(timedOut)
        {
            printf("  FAIL - %s: exceeded %.0fs wall-clock budget (%d iterations)\n",
                   name, MAX_WALL_SECONDS, itr);
            return false;
        }
    if(!solved)
        {
            printf("  FAIL - %s: ran %d iterations without reaching the goal\n", name, itr);
            return false;
        }
    if(leaked)
        {
            printf("  FAIL - %s: %.1f MB not freed after planner destruction (before %.1f MB, after %.1f MB)\n",
                   name, (freeBefore - freeAfter) / (1024.0 * 1024.0),
                   freeBefore / (1024.0 * 1024.0), freeAfter / (1024.0 * 1024.0));
            return false;
        }

    printf("  PASS - %s (solved in %d iterations)\n", name, itr);
    return true;
}

static inline bool wallClockExpired(const std::chrono::steady_clock::time_point& wallStart)
{
    double wallSec = std::chrono::duration<double>(std::chrono::steady_clock::now() - wallStart).count();
    return wallSec >= MAX_WALL_SECONDS;
}

// ============================================================================
// Family 1: d_pathToGoal_ solve signal, WITH graph_.updateVertices(). KPAX, PruneKPAX.
// Loop body copied from countingstars_sweep.cu's benchmarkKPAX.
// ============================================================================
template<typename PlannerType>
bool runSmokeTestPathToGoal(const char* name, float* h_initial, float* h_goal,
                            float* d_obstacles, uint numObstacles, int maxIterations)
{
    return runSmokeTestFramed(name, [&](bool& solved, bool& timedOut, int& itr, auto wallStart)
        {
            PlannerType planner;
            if(!checkCudaError(name)) return;

            planner.resetPlanner(h_initial, h_goal);
            if(!checkCudaError(name)) return;

            int zero = 0;
            while(itr < maxIterations)
                {
                    itr++;
                    planner.h_itr_++;

                    // Reset pathToGoal before each iteration so a solution found on a LATER
                    // iteration is not masked by one already found and left set from an earlier
                    // iteration -- matches benchmarkKPAX exactly.
                    cudaMemcpy(planner.d_pathToGoal_ptr_, &zero, sizeof(int), cudaMemcpyHostToDevice);
                    planner.h_pathToGoal_ = 0;

                    planner.propagateFrontier(d_obstacles, numObstacles);
                    planner.graph_.updateVertices();
                    planner.updateFrontier();   // internally syncs h_pathToGoal_ back from device

                    if(planner.h_pathToGoal_ != 0)
                        {
                            solved = true;
                            break;
                        }
                    if(planner.h_treeSize_ >= MAX_TREE_SIZE - 1) break;
                    if(wallClockExpired(wallStart)) { timedOut = true; break; }
                }
        });   // planner destroyed here, before runSmokeTestFramed's "after" memory reading
}

// ============================================================================
// Family 2: d_minCost_ solve signal, WITH graph_.updateVertices(). KinoPaxSTAR, KinoPaxSTARCleanCost.
// Loop body copied from countingstars_sweep.cu's benchmarkKinoPaxSTARCleanCost.
// ============================================================================
template<typename PlannerType>
bool runSmokeTestMinCostWithGraph(const char* name, float* h_initial, float* h_goal,
                                  float* d_obstacles, uint numObstacles, int maxIterations)
{
    return runSmokeTestFramed(name, [&](bool& solved, bool& timedOut, int& itr, auto wallStart)
        {
            PlannerType planner;
            if(!checkCudaError(name)) return;

            planner.resetPlanner(h_initial, h_goal);
            if(!checkCudaError(name)) return;

            while(itr < maxIterations)
                {
                    itr++;
                    planner.h_itr_++;

                    planner.propagateFrontier(d_obstacles, numObstacles);
                    planner.graph_.updateVertices();
                    planner.updateFrontier();

                    cudaMemcpy(&planner.h_minCost_, planner.d_minCost_ptr_, sizeof(float),
                              cudaMemcpyDeviceToHost);
                    if(planner.h_minCost_ < MAX_FLOAT)
                        {
                            solved = true;
                            break;
                        }
                    if(planner.h_treeSize_ >= MAX_TREE_SIZE - 1) break;
                    if(planner.h_propIterations_ == 0) break;
                    if(wallClockExpired(wallStart)) { timedOut = true; break; }
                }
        });
}

// ============================================================================
// KinoPaxPlus: d_minCost_ solve signal, NO graph_.updateVertices() (different graph type -- see
// header). h_propIterations_ is checked BEFORE updateFrontier(), unlike Family 2 above, because
// KinoPaxPlus can set it to 0 inside propagateFrontier() itself when the tree buffer is exhausted.
// Loop body copied from countingstars_sweep.cu's benchmarkKinoPaxPlus.
// ============================================================================
bool runSmokeTestKinoPaxPlus(const char* name, float* h_initial, float* h_goal,
                             float* d_obstacles, uint numObstacles, int maxIterations)
{
    return runSmokeTestFramed(name, [&](bool& solved, bool& timedOut, int& itr, auto wallStart)
        {
            KinoPaxPlus planner;
            if(!checkCudaError(name)) return;

            planner.resetPlanner(h_initial, h_goal);
            if(!checkCudaError(name)) return;

            while(itr < maxIterations)
                {
                    itr++;
                    planner.h_itr_++;

                    planner.propagateFrontier(d_obstacles, numObstacles);
                    if(planner.h_propIterations_ == 0) break;
                    planner.updateFrontier();

                    cudaMemcpy(&planner.h_minCost_, planner.d_minCost_ptr_, sizeof(float),
                              cudaMemcpyDeviceToHost);
                    if(planner.h_minCost_ < MAX_FLOAT)
                        {
                            solved = true;
                            break;
                        }
                    if(planner.h_treeSize_ >= MAX_TREE_SIZE - 1) break;
                    if(wallClockExpired(wallStart)) { timedOut = true; break; }
                }
        });
}

// ============================================================================
// CountingStars: d_minCost_ solve signal, NO graph_.updateVertices() (it is Graph-based but
// consumes nothing updateVertices() produces -- see countingstars_sweep.cu). The one planner this
// whole repo's Jetson coverage was missing before this change.
// Loop body copied from countingstars_sweep.cu's benchmarkCountingStars.
// ============================================================================
bool runSmokeTestCountingStars(const char* name, float* h_initial, float* h_goal,
                               float* d_obstacles, uint numObstacles, int maxIterations)
{
    return runSmokeTestFramed(name, [&](bool& solved, bool& timedOut, int& itr, auto wallStart)
        {
            CountingStars planner;
            if(!checkCudaError(name)) return;

            planner.resetPlanner(h_initial, h_goal);
            if(!checkCudaError(name)) return;

            while(itr < maxIterations)
                {
                    itr++;
                    planner.h_itr_++;

                    planner.propagateFrontier(d_obstacles, numObstacles);
                    planner.updateFrontier();

                    cudaMemcpy(&planner.h_minCost_, planner.d_minCost_ptr_, sizeof(float),
                              cudaMemcpyDeviceToHost);
                    if(planner.h_minCost_ < MAX_FLOAT)
                        {
                            solved = true;
                            break;
                        }
                    if(planner.h_treeSize_ >= MAX_TREE_SIZE - 1) break;
                    if(planner.h_propIterations_ == 0) break;
                    if(wallClockExpired(wallStart)) { timedOut = true; break; }
                }
        });
}

int main(void)
{
    const int MAX_ITERATIONS = 50;

    printf("=======================================================\n");
    printf("    JETSON SMOKE TEST\n");
    printf("=======================================================\n");
    printf("Running each planner for up to %d iterations (or %.0fs) on the Empty environment\n",
           MAX_ITERATIONS, MAX_WALL_SECONDS);
    printf("Purpose: verify every algorithm SOLVES, does not LEAK, and does not HANG\n");
    printf("=======================================================\n");

    printGPUMemory("initial");

    // Load empty environment (zero obstacles) -- an unobstructed start/goal is the right target for
    // a smoke test: any planner that cannot solve here has a real problem, not a hard environment.
    int numObstacles;
    float* d_obstacles;
    std::vector<float> obstacles = readObstaclesFromCSV(
        "../include/config/obstacles/empty/obstacles.csv", numObstacles, W_DIM);
    cudaMalloc(&d_obstacles, numObstacles * 2 * W_DIM * sizeof(float));
    cudaMemcpy(d_obstacles, obstacles.data(),
               numObstacles * 2 * W_DIM * sizeof(float), cudaMemcpyHostToDevice);
    printf("Loaded %d obstacles\n", numObstacles);

    // Start/goal DERIVED FROM CONFIG, not hardcoded. The old {10,8,5}->{80,95,90} pair matched only
    // the checked-in Model-3 config's W_MAX 100.0f; under any of the scripts/run_*.sh write_config
    // heredocs (W_MAX 1.0f) that goal sits 100x outside the workspace and every planner would
    // "solve" nothing while still printing PASS, because the old test never checked for a solution
    // at all. This is the exact pattern countingstars_sweep.cu's main() uses.
    float h_initial[SAMPLE_DIM] = {0};
    float h_goal[SAMPLE_DIM]    = {0};
    h_initial[0] = W_MIN + 0.1f  * W_SIZE;
    h_initial[1] = W_MIN + 0.08f * W_SIZE;
    h_initial[2] = W_MIN + 0.05f * W_SIZE;
    h_goal[0]    = W_MIN + 0.8f  * W_SIZE;
    h_goal[1]    = W_MIN + 0.95f * W_SIZE;
    h_goal[2]    = W_MIN + 0.9f  * W_SIZE;

    int passed = 0;
    int total  = 0;

    total++; passed += runSmokeTestPathToGoal<KPAX>("KPAX",
        h_initial, h_goal, d_obstacles, numObstacles, MAX_ITERATIONS);

    total++; passed += runSmokeTestPathToGoal<PruneKPAX>("PruneKPAX",
        h_initial, h_goal, d_obstacles, numObstacles, MAX_ITERATIONS);

    total++; passed += runSmokeTestMinCostWithGraph<KinoPaxSTAR>("KinoPaxSTAR",
        h_initial, h_goal, d_obstacles, numObstacles, MAX_ITERATIONS);

    // The arm CountingStars is actually measured against, in the slot a duplicated PruneKPAX run
    // used to occupy under the misleading label "KPAX_SpatialHash".
    total++; passed += runSmokeTestMinCostWithGraph<KinoPaxSTARCleanCost>("KinoPaxSTARCleanCost",
        h_initial, h_goal, d_obstacles, numObstacles, MAX_ITERATIONS);

    total++; passed += runSmokeTestKinoPaxPlus("KinoPaxPlus",
        h_initial, h_goal, d_obstacles, numObstacles, MAX_ITERATIONS);

    total++; passed += runSmokeTestCountingStars("CountingStars",
        h_initial, h_goal, d_obstacles, numObstacles, MAX_ITERATIONS);

    cudaFree(d_obstacles);

    printf("\n=======================================================\n");
    printf("    RESULTS: %d/%d PASSED\n", passed, total);
    printf("=======================================================\n");

    return (passed == total) ? 0 : 1;
}
