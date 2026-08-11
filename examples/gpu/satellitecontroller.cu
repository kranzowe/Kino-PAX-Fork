// =============================================================================
//  Satellite flag-capture demo for the KinoPaxSTARcostprune planner.
//
//  A 2D Clohessy-Wiltshire satellite (radial / in-track) starts at (1000, 1000) m
//  and must capture a flag at the origin while avoiding 10 defender satellites that
//  drift on CW dynamics and apply a random delta-V every 10-minute cycle.
//
//  Each cycle (receding horizon):
//    1. Build keep-out boxes around the current defender positions -> device.
//    2. Plan a full DV-optimal, safety-aware trajectory to the flag with CostPrune.
//    3. "Fly" the first SEGMENT_T seconds of that plan open-loop (instant, since sim).
//    4. Advance the defenders SEGMENT_T seconds (one random DV impulse + CW coast).
//    5. Repeat until the flag is captured or MAX_CYCLES is reached.
//
//  Compute runs on the GPU; nothing is plotted here. All state is streamed to CSV
//  under Data/SatellitePursuit/ for the MATLAB animation (viz/satellite_pursuit_anim.m).
//
//  Requires config.h MODEL 4 (the 2D CW model). Cost = dv_r^2 + dv_i^2 + W_SAFETY * safety.
// =============================================================================
#define _USE_MATH_DEFINES
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>

#include "config/config.h"
#include "helper/helper.cuh"
#include "planners/KinoPaxSTARcostprune.cuh"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace
{
// ---- Scenario constants (tunable) ----
constexpr int   NUM_DEFENDERS   = 10;           // number of defender satellites
constexpr float DEFENDER_DISK_R = 800.0f;       // initial spawn radius around the flag [m]
constexpr float R_KEEPOUT       = 50.0f;        // hard keep-out half-width per defender [m] (100 m box)
constexpr float DEFENDER_DV     = 0.10f;        // per-axis random DV bound, per cycle [m/s]
constexpr float DEFENDER_V0     = 0.05f;        // per-axis initial velocity bound [m/s]
constexpr float SEGMENT_T       = 600.0f;       // fly + defender-update horizon per cycle [s] (10 min)
constexpr float LOG_DT          = 20.0f;        // trajectory sampling resolution for the animation [s]
constexpr int   MAX_CYCLES      = 40;           // stop after this many cycles (or on capture)
constexpr float CAPTURE_R       = GOAL_THRESH;  // capture radius [m]
const std::string OUT_DIR       = "Data/SatellitePursuit";

// CW relative state = [x_radial, y_intrack, vx, vy].
struct State
{
    float x, y, vx, vy;
};

// Exact CW free-drift (coast, no control) by time t -- same state-transition matrix as the
// device propagator in statePropagator.cu (propagateAndCheckCW).
State cwCoast(const State& s0, float t)
{
    const float n  = MEAN_MOTION;
    const float sn = std::sin(n * t);
    const float cs = std::cos(n * t);
    const float nt = n * t;
    State s;
    s.x  = (4.0f - 3.0f * cs) * s0.x + (sn / n) * s0.vx + (2.0f / n) * (1.0f - cs) * s0.vy;
    s.y  = 6.0f * (sn - nt) * s0.x + s0.y - (2.0f / n) * (1.0f - cs) * s0.vx + (1.0f / n) * (4.0f * sn - 3.0f * nt) * s0.vy;
    s.vx = 3.0f * n * sn * s0.x + cs * s0.vx + 2.0f * sn * s0.vy;
    s.vy = -6.0f * n * (1.0f - cs) * s0.x - 2.0f * sn * s0.vx + (4.0f * cs - 3.0f) * s0.vy;
    return s;
}

float dist2D(float ax, float ay, float bx, float by)
{
    float dx = ax - bx, dy = ay - by;
    return std::sqrt(dx * dx + dy * dy);
}
}  // namespace

int main(int argc, char** argv)
{
    // --- Args: --seed N ---
    unsigned seed = 1;
    for(int i = 1; i < argc; ++i)
        {
            std::string a = argv[i];
            if(a == "--seed" && i + 1 < argc) seed = (unsigned)std::stoul(argv[++i]);
        }
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u11(-1.0f, 1.0f);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);

    // --- Scenario setup ---
    const float flagx = 0.0f, flagy = 0.0f;
    const float startx = 1000.0f, starty = 1000.0f;
    State sat = {startx, starty, 0.0f, 0.0f};

    std::vector<State> def(NUM_DEFENDERS);
    for(int i = 0; i < NUM_DEFENDERS; ++i)
        {
            float r  = DEFENDER_DISK_R * std::sqrt(u01(rng));  // uniform over the disk
            float th = 2.0f * (float)M_PI * u01(rng);
            def[i]   = {flagx + r * std::cos(th), flagy + r * std::sin(th), DEFENDER_V0 * u11(rng), DEFENDER_V0 * u11(rng)};
        }

    // --- Output directories + CSV headers ---
    std::filesystem::create_directories(OUT_DIR);
    std::filesystem::create_directories(OUT_DIR + "/plans");

    {
        std::ofstream f(OUT_DIR + "/meta.csv");
        f << "n,flag_x,flag_y,start_x,start_y,W_MIN,W_MAX,GOAL_THRESH,R_KEEPOUT,num_defenders,max_cycles,dt_segment_s,log_dt_s,"
             "safety_weight,dv_max\n";
        f << std::setprecision(9) << MEAN_MOTION << "," << flagx << "," << flagy << "," << startx << "," << starty << "," << W_MIN
          << "," << W_MAX << "," << GOAL_THRESH << "," << R_KEEPOUT << "," << NUM_DEFENDERS << "," << MAX_CYCLES << "," << SEGMENT_T
          << "," << LOG_DT << "," << W_SAFETY << "," << DV_MAX << "\n";
    }
    std::ofstream satf(OUT_DIR + "/sat_trajectory.csv");
    std::ofstream deff(OUT_DIR + "/defenders.csv");
    std::ofstream costf(OUT_DIR + "/costs.csv");
    satf << "cycle,t,x,y,vx,vy\n";
    deff << "cycle,t,id,x,y,vx,vy\n";
    costf << "cycle,plan_cost,dv_total,min_dist_to_defender,captured\n";
    satf << std::fixed << std::setprecision(4);
    deff << std::fixed << std::setprecision(4);
    costf << std::fixed << std::setprecision(6);

    // --- Planner (constructed once; plan() resets internally each cycle) ---
    KinoPaxSTARcostprune planner;
    planner.initializeRandomSeeds((int)seed);
    planner.h_acceptCap_      = 0.1f;   // CostPrune: cap Syclop exploration at 10%
    planner.h_costPruneExp_   = 1.0f;
    planner.h_costPruneFloor_ = 0.02f;

    float* d_obstacles = nullptr;
    cudaMalloc(&d_obstacles, (size_t)NUM_DEFENDERS * 2 * W_DIM * sizeof(float));
    std::vector<float> h_obstacles((size_t)NUM_DEFENDERS * 2 * W_DIM);

    float globalT = 0.0f;
    bool captured = false;

    for(int cycle = 0; cycle < MAX_CYCLES && !captured; ++cycle)
        {
            // 1) Defender keep-out boxes [xmin, ymin, xmax, ymax] -> device.
            for(int i = 0; i < NUM_DEFENDERS; ++i)
                {
                    h_obstacles[i * 2 * W_DIM + 0] = def[i].x - R_KEEPOUT;
                    h_obstacles[i * 2 * W_DIM + 1] = def[i].y - R_KEEPOUT;
                    h_obstacles[i * 2 * W_DIM + 2] = def[i].x + R_KEEPOUT;
                    h_obstacles[i * 2 * W_DIM + 3] = def[i].y + R_KEEPOUT;
                }
            cudaMemcpy(d_obstacles, h_obstacles.data(), h_obstacles.size() * sizeof(float), cudaMemcpyHostToDevice);

            // 2) Plan with CostPrune from the current satellite state to the flag.
            float h_initial[SAMPLE_DIM] = {0};
            float h_goal[SAMPLE_DIM]    = {0};
            h_initial[0] = sat.x;
            h_initial[1] = sat.y;
            h_initial[2] = sat.vx;
            h_initial[3] = sat.vy;
            h_goal[0]    = flagx;  // planner's goal test is position-only (distance over W_DIM)
            h_goal[1]    = flagy;
            planner.plan(h_initial, h_goal, d_obstacles, (uint)NUM_DEFENDERS);

            // 3) Reconstruct the plan. h_controlPathsToGoal_ is filled goal -> root; the buffer is
            //    zero-filled, so path length L = index of the first all-zero row (the start is not
            //    the origin, so the root row is never mistaken for zero-fill).
            const float* P       = planner.h_controlPathsToGoal_;
            const bool haveSol   = (planner.h_minCost_ < 0.5f * MAX_FLOAT);
            int L                = 0;
            if(haveSol)
                {
                    while(L < MAX_ITER)
                        {
                            const float* r = &P[L * SAMPLE_DIM];
                            bool allZero   = true;
                            for(int j = 0; j < SAMPLE_DIM; ++j)
                                if(r[j] != 0.0f) { allZero = false; break; }
                            if(allZero) break;
                            ++L;
                        }
                }

            // 4) Log the planned ghost path, forward order (root -> goal).
            {
                std::ofstream pf(OUT_DIR + "/plans/plan_cycle" + std::to_string(cycle) + ".csv");
                pf << "x,y,vx,vy,dv_r,dv_i\n" << std::fixed << std::setprecision(4);
                for(int k = L - 1; k >= 0; --k)
                    {
                        const float* r = &P[k * SAMPLE_DIM];
                        pf << r[0] << "," << r[1] << "," << r[2] << "," << r[3] << "," << r[4] << "," << r[5] << "\n";
                    }
            }

            // 5) Build the forward edge list (root -> goal). Path row k (k = 0..L-2) stores the
            //    DV impulse + duration of the edge that reached it from its parent (row k+1).
            struct Edge
            {
                State start;    // state AFTER the impulse, at Tstart
                float dur;      // coast duration [s]
                float Tstart;   // cumulative time at edge start [s]
            };
            std::vector<Edge> edges;
            float totalT  = 0.0f;
            float dvTotal = 0.0f;
            if(L >= 2)
                {
                    State st = sat;  // root
                    for(int k = L - 2; k >= 0; --k)
                        {
                            const float* r = &P[k * SAMPLE_DIM];
                            float dvr = r[4], dvi = r[5], dur = r[7];
                            st.vx += dvr;  // impulse at edge start
                            st.vy += dvi;
                            dvTotal += std::sqrt(dvr * dvr + dvi * dvi);
                            edges.push_back({st, dur, totalT});
                            st = cwCoast(st, dur);  // coast to edge end
                            totalT += dur;
                            if(totalT >= SEGMENT_T) break;  // enough edges to cover the segment
                        }
                }

            // 6) Fly open-loop for `horizon` seconds, sampling sat + defenders on a shared time grid.
            const float horizon = edges.empty() ? SEGMENT_T : std::min(SEGMENT_T, totalT);

            // State of the satellite at cycle-relative time tau in [0, horizon].
            auto satAt = [&](float tau) -> State
            {
                if(edges.empty()) return cwCoast(sat, tau);  // no usable plan: free drift (keeps sim alive)
                int e = 0;
                for(int i = 0; i < (int)edges.size(); ++i)
                    {
                        if(tau >= edges[i].Tstart) e = i;
                        else break;
                    }
                return cwCoast(edges[e].start, tau - edges[e].Tstart);
            };

            // Defenders: one random DV impulse per cycle, then coast. defBase = post-impulse states.
            std::vector<State> defBase(NUM_DEFENDERS);
            for(int i = 0; i < NUM_DEFENDERS; ++i)
                {
                    defBase[i]    = def[i];
                    defBase[i].vx += DEFENDER_DV * u11(rng);
                    defBase[i].vy += DEFENDER_DV * u11(rng);
                }

            float minDefDist = 1e30f;
            float captureT   = -1.0f;
            int nsteps       = (int)std::floor(horizon / LOG_DT + 1e-6f);
            for(int i = 0; i <= nsteps; ++i)
                {
                    float tt = (i < nsteps) ? (i * LOG_DT) : horizon;  // last sample exactly at horizon
                    State s  = satAt(tt);
                    satf << cycle << "," << (globalT + tt) << "," << s.x << "," << s.y << "," << s.vx << "," << s.vy << "\n";
                    for(int d = 0; d < NUM_DEFENDERS; ++d)
                        {
                            State dd = cwCoast(defBase[d], tt);
                            deff << cycle << "," << (globalT + tt) << "," << d << "," << dd.x << "," << dd.y << "," << dd.vx << ","
                                 << dd.vy << "\n";
                            minDefDist = std::min(minDefDist, dist2D(s.x, s.y, dd.x, dd.y));
                        }
                    if(!captured && dist2D(s.x, s.y, flagx, flagy) < CAPTURE_R)
                        {
                            captured  = true;
                            captureT  = globalT + tt;
                        }
                }

            // 7) Advance the true states to the end of the flown horizon.
            sat = satAt(horizon);
            for(int i = 0; i < NUM_DEFENDERS; ++i) def[i] = cwCoast(defBase[i], horizon);
            globalT += horizon;

            // 8) Per-cycle summary.
            float planCost = haveSol ? planner.h_minCost_ : -1.0f;
            costf << cycle << "," << planCost << "," << dvTotal << "," << minDefDist << "," << (captured ? 1 : 0) << "\n";
            std::cout << "[cycle " << cycle << "] plan_nodes=" << L << " cost=" << planCost << " dv=" << dvTotal
                      << " minDefDist=" << minDefDist << " dist_to_flag=" << dist2D(sat.x, sat.y, flagx, flagy)
                      << (captured ? "  <-- CAPTURED" : "") << std::endl;
            if(captured) std::cout << "Flag captured at t = " << captureT << " s." << std::endl;
        }

    if(!captured) std::cout << "Reached MAX_CYCLES without capture." << std::endl;

    cudaFree(d_obstacles);
    satf.close();
    deff.close();
    costf.close();
    std::cout << "Wrote CSVs to " << OUT_DIR << "/ (meta, sat_trajectory, defenders, costs, plans/plan_cycle*.csv)" << std::endl;
    return 0;
}
