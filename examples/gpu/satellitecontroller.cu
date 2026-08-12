// =============================================================================
//  Satellite flag-capture demo for the KinoPaxSTAR planner.
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
#include "planners/KinoPaxSTAR.cuh"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace
{
// ---- Scenario constants (tunable) ----
constexpr int   NUM_DEFENDERS   = 7;           // number of defender satellites
constexpr float DEFENDER_DISK_R = 800.0f;       // initial spawn radius around the flag [m]
constexpr float R_KEEPOUT       = 25.0f;        // hard keep-out half-width per defender [m] (100 m box)
constexpr float DEFENDER_WP_RMIN = 400.0f;      // defender waypoint annulus inner radius [m] (around flag)
constexpr float DEFENDER_WP_RMAX = 1200.0f;     // defender waypoint annulus outer radius [m] (around flag)
constexpr float DEFENDER_V0     = 0.05f;        // per-axis initial velocity bound [m/s]
constexpr float SEGMENT_T       = 60.0f;       // fly + defender-update horizon per cycle [s] (10 min)
constexpr float DEFENDER_TOF    = 3600.0f;    // defender time-of-flight to its waypoint [s] (tunable)
constexpr float DEFENDER_CHASE_PROB = 0.30f; // per defender, per cycle: probability of targeting the
                                             // satellite (chase) vs a random flag-vicinity waypoint
constexpr float THRUST_NOISE    = 0.20f;        // execution thrust error: each planned DV component is
                                                // perturbed by +/- this fraction at burn time (0.20 = 20%)
constexpr float LOG_DT          = 20.0f;        // trajectory sampling resolution for the animation [s]
constexpr float SMOOTH_DT       = 10.0f;        // sub-sample step for the smooth plan / fly-out curves [s]
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

// Re-propagate the FULL plan (all edges root->goal) into a dense, smooth (x,y) trajectory,
// sub-sampling each CW coast at SMOOTH_DT via cwCoast. If `noisy`, each planned DV component is
// perturbed by +/-THRUST_NOISE (one example open-loop fly-out of the whole plan). `rng` is used
// only when noisy; pass a stream separate from the main sim's so this never perturbs it.
void writeSmoothPlan(const State& root, const float* P, int L, bool noisy, std::mt19937& rng,
                     std::uniform_real_distribution<float>& u11, const std::string& fname)
{
    std::ofstream f(fname);
    f << "x,y\n" << std::fixed << std::setprecision(4);
    if(L < 2) return;                       // no plan (or root only): header only
    State st = root;
    f << st.x << "," << st.y << "\n";       // start point (the satellite's current state)
    for(int k = L - 2; k >= 0; --k)         // edges root->goal (same order as the fly loop)
        {
            const float* r = &P[k * SAMPLE_DIM];
            float dvr = r[4], dvi = r[5], dur = r[7];
            if(noisy)
                {
                    dvr *= (1.0f + THRUST_NOISE * u11(rng));
                    dvi *= (1.0f + THRUST_NOISE * u11(rng));
                }
            st.vx += dvr;                   // impulse at edge start
            st.vy += dvi;
            int   nsub = std::max(1, (int)std::ceil(dur / SMOOTH_DT));
            float sub  = dur / (float)nsub;
            for(int s = 0; s < nsub; ++s)   // sub-sample the coast for a smooth curve
                {
                    st = cwCoast(st, sub);
                    f << st.x << "," << st.y << "\n";
                }
        }
}

float dist2D(float ax, float ay, float bx, float by)
{
    float dx = ax - bx, dy = ay - by;
    return std::sqrt(dx * dx + dy * dy);
}

// Impulsive delta-V to reach target position (tx,ty) at time tof from state s, by inverting the
// CW position block:  v_req = Phi_rv(tof)^-1 * (r_target - Phi_rr(tof) * r0);  DV = v_req - v0.
// Phi_rv is 2x2, so the inverse is closed-form (singular only near whole-orbit tof).
void cwTargetingDV(const State& s, float tx, float ty, float tof, float& dvx, float& dvy)
{
    const float n  = MEAN_MOTION;
    const float sn = std::sin(n * tof);
    const float cs = std::cos(n * tof);
    const float nt = n * tof;

    // Phi_rr and Phi_rv 2x2 blocks of the CW state-transition matrix (rr01 = 0, rr11 = 1).
    const float rr00 = 4.0f - 3.0f * cs, rr10 = 6.0f * (sn - nt);
    const float rv00 = sn / n, rv01 = (2.0f / n) * (1.0f - cs);
    const float rv10 = -(2.0f / n) * (1.0f - cs), rv11 = (1.0f / n) * (4.0f * sn - 3.0f * nt);

    // b = r_target - Phi_rr * r0
    const float bx = tx - (rr00 * s.x);
    const float by = ty - (rr10 * s.x + s.y);

    // v_req = Phi_rv^-1 * b
    const float det = rv00 * rv11 - rv01 * rv10;
    if(std::fabs(det) < 1e-9f) { dvx = 0.0f; dvy = 0.0f; return; }  // near-singular tof: skip the burn
    const float vreqx = (rv11 * bx - rv01 * by) / det;
    const float vreqy = (-rv10 * bx + rv00 * by) / det;

    dvx = vreqx - s.vx;
    dvy = vreqy - s.vy;
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
    std::mt19937 flyoutRng(seed ^ 0x9E3779B9u);  // separate stream for the fly-out example noise,
                                                 // so it never perturbs the main sim's RNG

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
    costf << "cycle,plan_cost,dv_cost,safety_cost,time_cost,flight_time_s,dv_mps,min_dist_to_defender,captured\n";
    satf << std::fixed << std::setprecision(4);
    deff << std::fixed << std::setprecision(4);
    costf << std::fixed << std::setprecision(6);

    // --- Planner (constructed once; plan() resets internally each cycle) ---
    // KinoPaxSTAR: KPAX exploration + KinoPaxPlus cost + goal-progress pruning. Its
    // goal-progress tunables (h_maxRegression_/h_explorationBias_/h_goalBias_) default in
    // the constructor; left as-is here.
    KinoPaxSTAR planner;
    planner.initializeRandomSeeds((int)seed);

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

            // 4) Smooth ghost curves for this cycle's plan: nominal (planned DVs, no noise) and an
            //    example open-loop fly-out of the WHOLE plan under thrust noise. Both start at the
            //    satellite's current state. The fly-out draws from flyoutRng (a separate stream), so
            //    generating it never perturbs the main sim's RNG.
            writeSmoothPlan(sat, P, L, false, flyoutRng, u11, OUT_DIR + "/plans/plan_cycle" + std::to_string(cycle) + ".csv");
            writeSmoothPlan(sat, P, L, true, flyoutRng, u11, OUT_DIR + "/plans/flyout_cycle" + std::to_string(cycle) + ".csv");

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
            if(L >= 2)
                {
                    State st = sat;  // root
                    for(int k = L - 2; k >= 0; --k)
                        {
                            const float* r = &P[k * SAMPLE_DIM];
                            float dvr = r[4], dvi = r[5], dur = r[7];
                            // Execution thrust noise: the DV actually applied differs from the planned DV
                            // by up to +/-THRUST_NOISE per component. The planner planned from the clean
                            // base model; only this flown trajectory is perturbed, so the next replan (from
                            // the resulting off-nominal state) has to correct for the accumulated error.
                            float dvr_exec = dvr * (1.0f + THRUST_NOISE * u11(rng));
                            float dvi_exec = dvi * (1.0f + THRUST_NOISE * u11(rng));
                            st.vx += dvr_exec;  // noisy impulse at edge start
                            st.vy += dvi_exec;
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

            // Defenders: each cycle re-target a waypoint and burn (CW targeting) to arrive there in
            // DEFENDER_TOF seconds. With probability DEFENDER_CHASE_PROB a defender targets the
            // satellite's current position (chase); otherwise a random waypoint in the annulus
            // [DEFENDER_WP_RMIN, DEFENDER_WP_RMAX] around the flag (patrol). defBase = post-burn states.
            std::vector<State> defBase(NUM_DEFENDERS);
            for(int i = 0; i < NUM_DEFENDERS; ++i)
                {
                    float wx, wy;
                    if(u01(rng) < DEFENDER_CHASE_PROB)
                        {
                            // Chase: aim at where the satellite is right now (start of this cycle).
                            wx = sat.x;
                            wy = sat.y;
                        }
                    else
                        {
                            // Patrol: random waypoint, uniform over the flag-centered annulus.
                            float wr  = std::sqrt(u01(rng) * (DEFENDER_WP_RMAX * DEFENDER_WP_RMAX - DEFENDER_WP_RMIN * DEFENDER_WP_RMIN)
                                                  + DEFENDER_WP_RMIN * DEFENDER_WP_RMIN);
                            float wth = 2.0f * (float)M_PI * u01(rng);
                            wx = flagx + wr * std::cos(wth);
                            wy = flagy + wr * std::sin(wth);
                        }
                    float dvx, dvy;
                    cwTargetingDV(def[i], wx, wy, DEFENDER_TOF, dvx, dvy);
                    defBase[i]    = def[i];
                    defBase[i].vx += dvx;
                    defBase[i].vy += dvy;
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

            // 8) Cost breakdown over the FULL planned path, matching the device edgeCost accumulation:
            //    total = sum(dv_r^2 + dv_i^2) + W_SAFETY*sum(safety) + W_TIME*sum(dt).
            //    compTotal should reconcile with planner.h_minCost_ (a bug-check on the accounting;
            //    small float differences from summation order are expected). The root node's
            //    control/dt slots are zero, so it contributes nothing.
            float dvCost = 0.0f, safetyCost = 0.0f, timeCost = 0.0f, flightTime = 0.0f, dvMag = 0.0f;
            for(int k = 0; k < L; ++k)
                {
                    const float* r = &P[k * SAMPLE_DIM];
                    dvCost     += r[4] * r[4] + r[5] * r[5];
                    dvMag      += std::sqrt(r[4] * r[4] + r[5] * r[5]);
                    safetyCost += W_SAFETY * r[6];
                    timeCost   += W_TIME * r[7];
                    flightTime += r[7];
                }
            float compTotal = dvCost + safetyCost + timeCost;

            // 9) Per-cycle summary.
            float planCost = haveSol ? planner.h_minCost_ : -1.0f;
            costf << cycle << "," << planCost << "," << dvCost << "," << safetyCost << "," << timeCost << "," << flightTime << ","
                  << dvMag << "," << minDefDist << "," << (captured ? 1 : 0) << "\n";
            std::cout << "[cycle " << cycle << "] plan_nodes=" << L << " cost=" << planCost << "  | dv_cost=" << dvCost
                      << " safety_cost=" << safetyCost << " time_cost=" << timeCost << " (sum=" << compTotal << ")"
                      << "  | dv=" << dvMag << " m/s  flight=" << flightTime << " s"
                      << "  | minDefDist=" << minDefDist << " dist_to_flag=" << dist2D(sat.x, sat.y, flagx, flagy)
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
