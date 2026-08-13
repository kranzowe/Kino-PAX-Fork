#pragma once

/***************************/
/* 6D DOUBLE INTEGRATOR CONFIG  */
/***************************/

// #define MODEL 1
// #define COST_MODE 1  // path cost: 1 = control effort, 0 = workspace distance (see helper.cuh edgeCost)

// #define MAX_TREE_SIZE 300000
// #define MAX_ITER 300
// #define STEP_SIZE 0.1f
// #define MAX_PROPAGATION_DURATION 10

// #define GOAL_THRESH 0.05f

// #define STATE_DIM 6
// #define CONTROL_DIM 3
// #define SAMPLE_DIM (STATE_DIM + CONTROL_DIM + 1)

// #define W_DIM 3
// #define C_DIM 1
// #define V_DIM 3

// #define W_MIN 0.0f
// #define W_MAX 1.0f
// #define W_SIZE 1.0f

// #define C_MIN -M_PI
// #define C_MAX M_PI

// #define V_MIN -0.3f
// #define V_MAX 0.3f

// #define A_MIN -0.2f
// #define A_MAX 0.2f

// #define W_R1_LENGTH 8
// #define C_R1_LENGTH 1
// #define V_R1_LENGTH 4

// #define W_R2_LENGTH 2
// #define C_R2_LENGTH 1
// #define V_R2_LENGTH 2

// #define W_R1_SIZE ((W_MAX - W_MIN) / W_R1_LENGTH)
// #define C_R1_SIZE ((C_MAX - C_MIN) / C_R1_LENGTH)
// #define V_R1_SIZE ((V_MAX - V_MIN) / V_R1_LENGTH)

// #define W_R1_VOL (W_R1_SIZE * W_R1_SIZE * W_R1_SIZE)

// #define NUM_R1_REGIONS (W_R1_LENGTH * W_R1_LENGTH * W_R1_LENGTH * V_R1_LENGTH * V_R1_LENGTH * V_R1_LENGTH)
// #define NUM_R2_REGIONS (NUM_R1_REGIONS * W_R2_LENGTH * W_R2_LENGTH * W_R2_LENGTH * V_R2_LENGTH * V_R2_LENGTH * V_R2_LENGTH)
// #define NUM_R2_PER_R1 W_R2_LENGTH *W_R2_LENGTH *W_R2_LENGTH *V_R2_LENGTH *V_R2_LENGTH *V_R2_LENGTH
// #define NUM_R1_REGIONS_KERNEL1 1024
// #define NUM_PARTIAL_SUMS 1024

// #define EPSILON 1e-2f
// #define VERBOSE 1

// // --- UNICYCLE MODEL: MODEL 0 ---
// #define UNI_MIN_STEERING -M_PI / 2
// #define UNI_MAX_STEERING M_PI / 2
// #define UNI_MIN_DT 0.1f
// #define UNI_MAX_DT 2.0f
// #define UNI_LENGTH 1.0f

// // --- DUBINS AIRPLANE: MODEL 2 ---
// #define DUBINS_AIRPLANE_MIN_PR (-M_PI / 4)
// #define DUBINS_AIRPLANE_MAX_PR (M_PI / 4)
// #define DUBINS_AIRPLANE_MIN_YR (-M_PI / 4)
// #define DUBINS_AIRPLANE_MAX_YR (M_PI / 4)
// #define DUBINS_AIRPLANE_MIN_YAW -M_PI
// #define DUBINS_AIRPLANE_MAX_YAW M_PI
// #define DUBINS_AIRPLANE_MIN_PITCH -M_PI / 3
// #define DUBINS_AIRPLANE_MAX_PITCH M_PI / 3

// // --- NON LINEAR QUAD: MODEL 3 ---
// #define QUAD_MIN_Zc -2.0f
// #define QUAD_MAX_Zc 2.0f
// #define QUAD_MIN_Lc -2.0f
// #define QUAD_MAX_Lc 2.0f
// #define QUAD_MIN_Mc -2.0f
// #define QUAD_MAX_Mc 2.0f
// #define QUAD_MIN_Nc -2.0f
// #define QUAD_MAX_Nc 2.0f
// #define QUAD_MIN_YAW -M_PI
// #define QUAD_MAX_YAW M_PI
// #define QUAD_MIN_PITCH -M_PI / 3
// #define QUAD_MAX_PITCH M_PI / 3
// #define QUAD_MIN_ROLL -M_PI / 3
// #define QUAD_MAX_ROLL M_PI / 3
// #define QUAD_MIN_ANGLE_RATE -30.0f
// #define QUAD_MAX_ANGLE_RATE 30.0f
// #define NU 10e-3f
// #define MU 2e-6f
// #define KM 0.03f
// #define IX 1.0f
// #define IY 1.0f
// #define IZ 2.0f
// #define GRAVITY -9.81f
// #define MASS 1.0f
// #define MASS_INV 1.0f / MASS

/***************************/
/* DUBINS AIRPLANE CONFIG  */
/***************************/

// #define MODEL 2

// #define MAX_TREE_SIZE 200000
// #define MAX_ITER 300
// #define STEP_SIZE 0.1f
// #define MAX_PROPAGATION_DURATION 10

// #define GOAL_THRESH 0.05f

// #define STATE_DIM 6
// #define CONTROL_DIM 3
// #define SAMPLE_DIM (STATE_DIM + CONTROL_DIM + 1)

// #define W_DIM 3
// #define C_DIM 2
// #define V_DIM 1

// #define W_MIN 0.0f
// #define W_MAX 1.0f
// #define W_SIZE 1.0f

// #define C_MIN -M_PI
// #define C_MAX M_PI

// #define V_MIN 0.0f
// #define V_MAX 0.3f

// #define A_MIN -0.3f
// #define A_MAX 0.3f

// #define W_R1_LENGTH 8
// #define C_R1_LENGTH 8
// #define V_R1_LENGTH 1

// #define W_R2_LENGTH 2
// #define C_R2_LENGTH 3
// #define V_R2_LENGTH 1

// #define W_R1_SIZE ((W_MAX - W_MIN) / W_R1_LENGTH)
// #define C_R1_SIZE ((C_MAX - C_MIN) / C_R1_LENGTH)
// #define V_R1_SIZE ((V_MAX - V_MIN) / V_R1_LENGTH)

// #define W_R1_VOL (W_R1_SIZE * W_R1_SIZE * W_R1_SIZE)

// #define NUM_R1_REGIONS (W_R1_LENGTH * W_R1_LENGTH * W_R1_LENGTH * C_R1_LENGTH * C_R1_LENGTH * V_R1_LENGTH)
// #define NUM_R2_REGIONS (NUM_R1_REGIONS * W_R2_LENGTH * W_R2_LENGTH * W_R2_LENGTH * C_R2_LENGTH * C_R2_LENGTH * V_R2_LENGTH)
// #define NUM_R2_PER_R1 W_R2_LENGTH *W_R2_LENGTH *W_R2_LENGTH *C_R2_LENGTH *C_R2_LENGTH *V_R2_LENGTH
// #define NUM_R1_REGIONS_KERNEL1 1024
// #define NUM_PARTIAL_SUMS 1024

// #define EPSILON 1e-2f
// #define VERBOSE 1

// // --- UNICYCLE MODEL: MODEL 0 ---
// #define UNI_MIN_STEERING -M_PI / 2
// #define UNI_MAX_STEERING M_PI / 2
// #define UNI_MIN_DT 0.1f
// #define UNI_MAX_DT 2.0f
// #define UNI_LENGTH 1.0f

// // --- DUBINS AIRPLANE: MODEL 2 ---
// #define DUBINS_AIRPLANE_MIN_PR (-M_PI / 4)
// #define DUBINS_AIRPLANE_MAX_PR (M_PI / 4)
// #define DUBINS_AIRPLANE_MIN_YR (-M_PI / 4)
// #define DUBINS_AIRPLANE_MAX_YR (M_PI / 4)
// #define DUBINS_AIRPLANE_MIN_YAW -M_PI
// #define DUBINS_AIRPLANE_MAX_YAW M_PI
// #define DUBINS_AIRPLANE_MIN_PITCH -M_PI / 3
// #define DUBINS_AIRPLANE_MAX_PITCH M_PI / 3

// // --- NON LINEAR QUAD: MODEL 3 ---
// #define QUAD_MIN_Zc -2.0f
// #define QUAD_MAX_Zc 2.0f
// #define QUAD_MIN_Lc -2.0f
// #define QUAD_MAX_Lc 2.0f
// #define QUAD_MIN_Mc -2.0f
// #define QUAD_MAX_Mc 2.0f
// #define QUAD_MIN_Nc -2.0f
// #define QUAD_MAX_Nc 2.0f
// #define QUAD_MIN_YAW -M_PI
// #define QUAD_MAX_YAW M_PI
// #define QUAD_MIN_PITCH -M_PI / 3
// #define QUAD_MAX_PITCH M_PI / 3
// #define QUAD_MIN_ROLL -M_PI / 3
// #define QUAD_MAX_ROLL M_PI / 3
// #define QUAD_MIN_ANGLE_RATE -30.0f
// #define QUAD_MAX_ANGLE_RATE 30.0f
// #define NU 10e-3f
// #define MU 2e-6f
// #define KM 0.03f
// #define IX 1.0f
// #define IY 1.0f
// #define IZ 2.0f
// #define GRAVITY -9.81f
// #define MASS 1.0f
// #define MASS_INV 1.0f / MASS

/*******************************************/
/* 2D CLOHESSY-WILTSHIRE SATELLITE CONFIG   */
/*  (MODEL 4) flag-capture demo driving     */
/*  KinoPaxSTARcostprune. Units: meters, s. */
/*  State  = [x_radial, y_intrack, vx, vy]  */
/*  Control= [dv_radial, dv_intrack, safety]*/
/*  Sample = [state | control | dt]         */
/*******************************************/

#define MODEL 4

#define MAX_TREE_SIZE 1000000
#define MAX_ITER 300
#define MAX_ITER_REKINO 20000
#define STEP_SIZE 15.0f              // collision/safety sub-step [s]; CW coast is exact via STM
#define MAX_PROPAGATION_DURATION 20  // max sub-steps per edge -> edge up to 300 s

#define GOAL_THRESH 40.0f           // planner solution tolerance to the flag [m]

#define STATE_DIM 4
#define CONTROL_DIM 3                // dv_r, dv_i, safetyPenalty (slot 3 = precomputed safety integral)
#define SAMPLE_DIM (STATE_DIM + CONTROL_DIM + 1)

#define W_DIM 2
#define C_DIM 0
#define V_DIM 2

// --- Orbital dynamics: LEO, ~90 min period ---
#define MEAN_MOTION 0.0011636f       // n = 2*pi / 5400 s  [rad/s]

// --- Workspace (signed, flag-centered) [m] ---
#define W_MIN -2000.0f
#define W_MAX 2000.0f
#define W_SIZE (W_MAX - W_MIN)

// --- Hard node-rejection box [m]: propagateAndCheckCW rejects any node that leaves this
// radial / in-track extent, marking the motion invalid (same effect as a collision). Kept
// inside the region grid (W_MIN/W_MAX = +/-3000), so the reachable region is comfortably
// covered by the Syclop bins (no edge-cell clamping). ---
#define NODE_RADIAL_LIMIT 2000.0f     // +/- 2000 m radial
#define NODE_INTRACK_LIMIT 2000.0f    // +/- 2000 m in-track

// --- Attitude placeholders (unused; C_DIM = 0). Kept defined so shared macros compile. ---
#define C_MIN -M_PI
#define C_MAX M_PI

// --- CW relative velocity bounds [m/s] ---
#define V_MIN -5.0f
#define V_MAX 5.0f

// --- Impulsive delta-V bound per edge, per axis [m/s] (the sampled control) ---
// The min-Delta-V transfer from (1000,1000)->origin needs ~2 m/s of departure burn; the DV^2
// cost favors splitting it across edges, so +/-1.0 per impulse composes it in a few burns.
#define DV_MIN -1.0f
#define DV_MAX 1.0f
// Generic accel-bound aliases: referenced by the other models' propagators (still compiled).
#define A_MIN DV_MIN
#define A_MAX DV_MAX

// --- Safety weight: edgeCost = dv_r^2 + dv_i^2 + W_SAFETY * sum_substep sum_def (STEP_SIZE / dist_to_center) ---
#define W_SAFETY 5.0f
#define W_TIME 0.002f

// --- Syclop region grid: W_DIM=2 spatial, V_DIM=2 velocity, no attitude ---
#define W_R1_LENGTH 20
#define C_R1_LENGTH 1
#define V_R1_LENGTH 14

#define W_R2_LENGTH 2
#define C_R2_LENGTH 1
#define V_R2_LENGTH 2

#define W_R1_SIZE ((W_MAX - W_MIN) / W_R1_LENGTH)
#define C_R1_SIZE ((C_MAX - C_MIN) / C_R1_LENGTH)
#define V_R1_SIZE ((V_MAX - V_MIN) / V_R1_LENGTH)

#define W_R1_VOL (W_R1_SIZE * W_R1_SIZE)

// R1 = W_R1^W_DIM * V_R1^V_DIM  (C_DIM = 0). R2 refines each R1 by W_R2^W_DIM * V_R2^V_DIM.
#define NUM_R1_REGIONS (W_R1_LENGTH * W_R1_LENGTH * V_R1_LENGTH * V_R1_LENGTH)
#define NUM_R2_PER_R1 (W_R2_LENGTH * W_R2_LENGTH * V_R2_LENGTH * V_R2_LENGTH)
#define NUM_R2_REGIONS (NUM_R1_REGIONS * NUM_R2_PER_R1)
#define NUM_R1_REGIONS_KERNEL1 1024
#define NUM_PARTIAL_SUMS 1024

#define EPSILON 1e-2f
#define MAX_FLOAT 1e38f
#define VERBOSE 0

#define KINOPAXPLUS_PARENT_CHAIN_PRUNING 1

#if 0  // ===== disabled MODEL 3 (nonlinear quad) core block: superseded by MODEL 4 above =====
/***************************/
/* NON LINEAR QUAD CONFIG  */
/***************************/

#define MODEL 3

#define MAX_TREE_SIZE 400000
#define MAX_ITER 300
#define MAX_ITER_REKINO 20000
#define STEP_SIZE 0.1f
#define MAX_PROPAGATION_DURATION 10

#define GOAL_THRESH 5.0f

#define STATE_DIM 12
#define CONTROL_DIM 4
#define SAMPLE_DIM (STATE_DIM + CONTROL_DIM + 1)

#define W_DIM 3
#define C_DIM 3
#define V_DIM 3

#define W_MIN 0.0f
#define W_MAX 100.0f
#define W_SIZE 100.0f

#define C_MIN -M_PI
#define C_MAX M_PI

#define V_MIN -30.0f
#define V_MAX 30.0f

#define A_MIN -30.0f
#define A_MAX 30.0f

#define W_R1_LENGTH 8
#define C_R1_LENGTH 2
#define V_R1_LENGTH 2

#define W_R2_LENGTH 2
#define C_R2_LENGTH 1
#define V_R2_LENGTH 2

#define W_R1_SIZE ((W_MAX - W_MIN) / W_R1_LENGTH)
#define C_R1_SIZE ((C_MAX - C_MIN) / C_R1_LENGTH)
#define V_R1_SIZE ((V_MAX - V_MIN) / V_R1_LENGTH)

#define W_R1_VOL (W_R1_SIZE * W_R1_SIZE * W_R1_SIZE)

#define NUM_R1_REGIONS \
    (W_R1_LENGTH * W_R1_LENGTH * W_R1_LENGTH * C_R1_LENGTH * C_R1_LENGTH * C_R1_LENGTH * V_R1_LENGTH * V_R1_LENGTH * V_R1_LENGTH)
#define NUM_R2_REGIONS                                                                                                                \
    (NUM_R1_REGIONS * W_R2_LENGTH * W_R2_LENGTH * W_R2_LENGTH * C_R2_LENGTH * C_R2_LENGTH * C_R2_LENGTH * V_R2_LENGTH * V_R2_LENGTH * \
     V_R2_LENGTH)
#define NUM_R2_PER_R1 W_R2_LENGTH *W_R2_LENGTH *W_R2_LENGTH *C_R2_LENGTH *C_R2_LENGTH *C_R2_LENGTH *V_R2_LENGTH *V_R2_LENGTH *V_R2_LENGTH
#define NUM_R1_REGIONS_KERNEL1 1024
#define NUM_PARTIAL_SUMS 1024

#define EPSILON 1e-2f
#define MAX_FLOAT 1e38f
#define VERBOSE 1

// Set to 1 to enable parent-chain validation in pruning (default behavior)
// Set to 0 to only check the node itself, not its ancestors
#define KINOPAXPLUS_PARENT_CHAIN_PRUNING 1

#endif  // ===== end disabled MODEL 3 (nonlinear quad) core block =====

// --- UNICYCLE MODEL: MODEL 0 ---
#define UNI_MIN_STEERING -M_PI / 2
#define UNI_MAX_STEERING M_PI / 2
#define UNI_MIN_DT 0.1f
#define UNI_MAX_DT 2.0f
#define UNI_LENGTH 1.0f

// --- DUBINS AIRPLANE: MODEL 2 ---
#define DUBINS_AIRPLANE_MIN_PR (-M_PI / 4)
#define DUBINS_AIRPLANE_MAX_PR (M_PI / 4)
#define DUBINS_AIRPLANE_MIN_YR (-M_PI / 4)
#define DUBINS_AIRPLANE_MAX_YR (M_PI / 4)
#define DUBINS_AIRPLANE_MIN_YAW -M_PI
#define DUBINS_AIRPLANE_MAX_YAW M_PI
#define DUBINS_AIRPLANE_MIN_PITCH -M_PI / 3
#define DUBINS_AIRPLANE_MAX_PITCH M_PI / 3

// --- NON LINEAR QUAD: MODEL 3 ---
#define QUAD_MIN_Zc 0.0f
#define QUAD_MAX_Zc 30.0f
#define QUAD_MIN_Lc -M_PI
#define QUAD_MAX_Lc M_PI
#define QUAD_MIN_Mc -M_PI
#define QUAD_MAX_Mc M_PI
#define QUAD_MIN_Nc -M_PI
#define QUAD_MAX_Nc M_PI
#define QUAD_MIN_YAW -M_PI
#define QUAD_MAX_YAW M_PI
#define QUAD_MIN_PITCH -M_PI
#define QUAD_MAX_PITCH M_PI
#define QUAD_MIN_ROLL -M_PI
#define QUAD_MAX_ROLL M_PI
#define QUAD_MIN_ANGLE_RATE -30.0f
#define QUAD_MAX_ANGLE_RATE 30.0f
#define NU 10e-3f
#define MU 2e-6f
#define KM 0.03f
#define IX 1.0f
#define IY 1.0f
#define IZ 2.0f
#define GRAVITY -9.81f
#define MASS 1.0f
#define MASS_INV 1.0f / MASS