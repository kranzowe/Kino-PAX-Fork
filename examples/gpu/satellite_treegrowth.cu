// =============================================================================
//  One-off tree-growth probe for the satellite flag-capture scenario.
//
//  Sets up the same cycle-0 scenario as satellitecontroller (satellite at
//  (1000,1000), flag at origin, 10 defenders in an 800 m disk), runs KinoPaxSTAR
//  ONCE, and dumps the whole search tree (with each node's birth iteration) plus
//  the solution path and obstacles. Feed the CSVs to viz/satellite_tree_growth.m
//  for an animated tree-growth plot.
//
//  Build:  make satellite_treegrowth -j
//  Run:    ./satellite_treegrowth [--seed N]
//  Out:    Data/SatelliteTree/{tree,obstacles,solution,meta}.csv
// =============================================================================
#define _USE_MATH_DEFINES
#include <iostream>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>

#include "config/config.h"
#include "helper/helper.cuh"
#include "planners/KinoPaxSTAR.cuh"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main(int argc, char** argv)
{
    unsigned seed = 1;
    for(int i = 1; i < argc; ++i)
        {
            std::string a = argv[i];
            if(a == "--seed" && i + 1 < argc) seed = (unsigned)std::stoul(argv[++i]);
        }

    // --- Scenario (matches satellitecontroller cycle 0) ---
    const int   NUM_DEFENDERS = 10;
    const float DISK_R = 800.0f, R_KEEPOUT = 50.0f, V0 = 0.05f;
    const float flagx = 0.0f, flagy = 0.0f, startx = 1000.0f, starty = 1000.0f;
    const std::string OUT = "Data/SatelliteTree";
    (void)V0;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);

    // Defenders in a disk around the flag -> keep-out boxes [xmin,ymin,xmax,ymax].
    std::vector<float> h_obstacles((size_t)NUM_DEFENDERS * 2 * W_DIM);
    for(int i = 0; i < NUM_DEFENDERS; ++i)
        {
            float r  = DISK_R * std::sqrt(u01(rng));
            float th = 2.0f * (float)M_PI * u01(rng);
            float dx = flagx + r * std::cos(th), dy = flagy + r * std::sin(th);
            h_obstacles[i * 2 * W_DIM + 0] = dx - R_KEEPOUT;
            h_obstacles[i * 2 * W_DIM + 1] = dy - R_KEEPOUT;
            h_obstacles[i * 2 * W_DIM + 2] = dx + R_KEEPOUT;
            h_obstacles[i * 2 * W_DIM + 3] = dy + R_KEEPOUT;
        }
    float* d_obstacles = nullptr;
    cudaMalloc(&d_obstacles, h_obstacles.size() * sizeof(float));
    cudaMemcpy(d_obstacles, h_obstacles.data(), h_obstacles.size() * sizeof(float), cudaMemcpyHostToDevice);

    float h_initial[SAMPLE_DIM] = {0}, h_goal[SAMPLE_DIM] = {0};
    h_initial[0] = startx;
    h_initial[1] = starty;
    h_goal[0]    = flagx;
    h_goal[1]    = flagy;

    // --- Single KinoPaxSTAR run ---
    KinoPaxSTAR planner;
    planner.initializeRandomSeeds((int)seed);
    planner.plan(h_initial, h_goal, d_obstacles, (uint)NUM_DEFENDERS);

    // --- Dump the tree ---
    std::filesystem::create_directories(OUT);
    int n                          = (int)planner.h_treeSize_;
    thrust::host_vector<float> S   = planner.d_treeSamples_;
    thrust::host_vector<int>   Par = planner.d_treeSamplesParentIdxs_;
    thrust::host_vector<float> Cst = planner.d_treeSampleCosts_;
    thrust::host_vector<int>   Itr = planner.d_iterations_;

    {
        std::ofstream f(OUT + "/tree.csv");
        f << "idx,x,y,vx,vy,parent,cost,iter\n" << std::fixed << std::setprecision(4);
        for(int i = 0; i < n; ++i)
            f << i << "," << S[i * SAMPLE_DIM + 0] << "," << S[i * SAMPLE_DIM + 1] << "," << S[i * SAMPLE_DIM + 2] << ","
              << S[i * SAMPLE_DIM + 3] << "," << Par[i] << "," << Cst[i] << "," << Itr[i] << "\n";
    }
    {
        std::ofstream f(OUT + "/obstacles.csv");
        f << "xmin,ymin,xmax,ymax\n" << std::fixed << std::setprecision(4);
        for(int i = 0; i < NUM_DEFENDERS; ++i)
            f << h_obstacles[i * 2 * W_DIM + 0] << "," << h_obstacles[i * 2 * W_DIM + 1] << "," << h_obstacles[i * 2 * W_DIM + 2]
              << "," << h_obstacles[i * 2 * W_DIM + 3] << "\n";
    }
    {
        // Solution path root->goal (real tree nodes only; the flag is NOT appended).
        std::ofstream f(OUT + "/solution.csv");
        f << "x,y,vx,vy\n" << std::fixed << std::setprecision(4);
        if(planner.h_minCost_ < 0.5f * MAX_FLOAT)
            {
                const float* P = planner.h_controlPathsToGoal_;
                int L          = 0;
                while(L < MAX_ITER)
                    {
                        const float* r = &P[L * SAMPLE_DIM];
                        bool z         = true;
                        for(int j = 0; j < SAMPLE_DIM; ++j)
                            if(r[j] != 0.0f) { z = false; break; }
                        if(z) break;
                        ++L;
                    }
                for(int k = L - 1; k >= 0; --k)
                    {
                        const float* r = &P[k * SAMPLE_DIM];
                        f << r[0] << "," << r[1] << "," << r[2] << "," << r[3] << "\n";
                    }
            }
    }
    {
        std::ofstream f(OUT + "/meta.csv");
        f << "flag_x,flag_y,start_x,start_y,W_MIN,W_MAX,GOAL_THRESH,R_KEEPOUT,num_defenders,tree_size,iterations,min_cost\n";
        f << std::setprecision(6) << flagx << "," << flagy << "," << startx << "," << starty << "," << W_MIN << "," << W_MAX << ","
          << GOAL_THRESH << "," << R_KEEPOUT << "," << NUM_DEFENDERS << "," << n << "," << planner.h_itr_ << "," << planner.h_minCost_
          << "\n";
    }

    std::cout << "Tree dump: " << n << " nodes, " << planner.h_itr_ << " iters, minCost=" << planner.h_minCost_ << " -> " << OUT
              << "/  (reached goal: " << (planner.h_minCost_ < 0.5f * MAX_FLOAT ? "yes" : "no") << ")" << std::endl;
    cudaFree(d_obstacles);
    return 0;
}
