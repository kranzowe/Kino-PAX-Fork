
#include "statePropagator/statePropagator.cuh"

__device__ bool propagateAndCheck(float* x0, float* x1, curandState* seed, float* obstacles, int obstaclesCount)
{
    PropagateAndCheckFunc func = getPropagateAndCheckFunc();
    return func ? func(x0, x1, seed, obstacles, obstaclesCount) : false;
}

/***************************/
/* UNICYCLE PROPAGATION FUNCTION */
/***************************/
__device__ bool propagateAndCheckUnicycle(float* x0, float* x1, curandState* seed, float* obstacles, int obstaclesCount)
{
    float a                 = A_MIN + curand_uniform(seed) * (A_MAX - A_MIN);
    float steering          = UNI_MIN_STEERING + curand_uniform(seed) * (UNI_MAX_STEERING - UNI_MIN_STEERING);
    float duration          = UNI_MIN_DT + curand_uniform(seed) * (UNI_MAX_DT - UNI_MIN_DT);
    int propagationDuration = 1 + (int)(curand_uniform(seed) * (MAX_PROPAGATION_DURATION));

    float x     = x0[0];
    float y     = x0[1];
    float theta = x0[2];
    float v     = x0[3];

    float cosTheta, sinTheta, tanSteering;
    float bbMin[W_DIM], bbMax[W_DIM];

    bool motionValid = true;
    for(int i = 0; i < propagationDuration; i++)
        {
            float x0State[W_DIM] = {x, y};
            cosTheta             = cos(theta);
            sinTheta             = sin(theta);
            tanSteering          = tan(steering);

            // --- State Propagation ---
            x += v * cosTheta * STEP_SIZE;
            y += v * sinTheta * STEP_SIZE;
            theta += (v / UNI_LENGTH) * tanSteering * STEP_SIZE;
            v += a * STEP_SIZE;
            float x1State[W_DIM] = {x, y};

            // --- Workspace Limit Check ---
            if(x < 0 || x > W_SIZE || y < 0 || y > W_SIZE)
                {
                    motionValid = false;
                    break;
                }

            // --- Obstacle Collision Check ---
            for(int d = 0; d < W_DIM; d++)
                {
                    if(x0State[d] > x1State[d])
                        {
                            bbMin[d] = x1State[d];
                            bbMax[d] = x0State[d];
                        }
                    else
                        {
                            bbMin[d] = x0State[d];
                            bbMax[d] = x1State[d];
                        }
                }

            motionValid = motionValid && isMotionValid(x0State, x1State, bbMin, bbMax, obstacles, obstaclesCount);
            if(!motionValid) break;
        }

    x1[0] = x, x1[1] = y, x1[2] = theta, x1[3] = v, x1[4] = a, x1[5] = steering, x1[6] = duration;
    return motionValid;
}

/***************************/
/* DOUBLE INTEGRATOR PROPAGATION FUNCTION */
/***************************/
__device__ bool propagateAndCheckDoubleIntRungeKutta(float* x0, float* x1, curandState* seed, float* obstacles, int obstaclesCount)
{
    float ax                = A_MIN + curand_uniform(seed) * (A_MAX - A_MIN);
    float ay                = A_MIN + curand_uniform(seed) * (A_MAX - A_MIN);
    float az                = A_MIN + curand_uniform(seed) * (A_MAX - A_MIN);
    int propagationDuration = 1 + (int)(curand_uniform(seed) * (MAX_PROPAGATION_DURATION));

    float x  = x0[0];
    float y  = x0[1];
    float z  = x0[2];
    float vx = x0[3];
    float vy = x0[4];
    float vz = x0[5];

    bool motionValid = true;
    float bbMin[W_DIM], bbMax[W_DIM];
    for(int i = 0; i < propagationDuration; i++)
        {
            float x0State[W_DIM] = {x, y, z};

            // --- State Propagation. 4th order Runge Kutta ---
            x += (vx + (vx + 2 * (vx + ax * STEP_SIZE / 2) + (vx + ax * STEP_SIZE))) * STEP_SIZE / 6;
            y += (vy + (vy + 2 * (vy + ay * STEP_SIZE / 2) + (vy + ay * STEP_SIZE))) * STEP_SIZE / 6;
            z += (vz + (vz + 2 * (vz + az * STEP_SIZE / 2) + (vz + az * STEP_SIZE))) * STEP_SIZE / 6;
            vx += (ax + 2 * ax + 2 * ax + ax) * STEP_SIZE / 6;
            vy += (ay + 2 * ay + 2 * ay + ay) * STEP_SIZE / 6;
            vz += (az + 2 * az + 2 * az + az) * STEP_SIZE / 6;

            // --- Dyanmics Validity Check ---
            if(vx < V_MIN || vx > V_MAX || vy < V_MIN || vy > V_MAX || vz < V_MIN || vz > V_MAX)
                {
                    motionValid = false;
                    break;
                }

            float x1State[W_DIM] = {x, y, z};

            // --- Workspace Limit Check ---
            if(x < 0 || x > W_SIZE || y < 0 || y > W_SIZE || z < 0 || z > W_SIZE)
                {
                    motionValid = false;
                    break;
                }

            // --- Obstacle Collision Check ---
            for(int d = 0; d < W_DIM; d++)
                {
                    if(x0State[d] > x1State[d])
                        {
                            bbMin[d] = x1State[d];
                            bbMax[d] = x0State[d];
                        }
                    else
                        {
                            bbMin[d] = x0State[d];
                            bbMax[d] = x1State[d];
                        }
                }

            motionValid = motionValid && isMotionValid(x0State, x1State, bbMin, bbMax, obstacles, obstaclesCount);
            if(!motionValid) break;
        }

    x1[0] = x, x1[1] = y, x1[2] = z, x1[3] = vx, x1[4] = vy, x1[5] = vz, x1[6] = ax, x1[7] = ay, x1[8] = az,
    x1[9] = STEP_SIZE * propagationDuration;
    return motionValid;
}

/***************************/
/* DUBINS AIRPLANE PROPAGATION FUNCTION */
/***************************/
__device__ bool propagateAndCheckDubinsAirplaneRungeKutta(float* x0, float* x1, curandState* seed, float* obstacles, int obstaclesCount)
{
    float a                 = A_MIN + curand_uniform(seed) * (A_MAX - A_MIN);
    float yawRate           = DUBINS_AIRPLANE_MIN_YR + curand_uniform(seed) * (DUBINS_AIRPLANE_MAX_YR - DUBINS_AIRPLANE_MIN_YR);
    float pitchRate         = DUBINS_AIRPLANE_MIN_PR + curand_uniform(seed) * (DUBINS_AIRPLANE_MAX_PR - DUBINS_AIRPLANE_MIN_PR);
    int propagationDuration = 1 + (int)(curand_uniform(seed) * (MAX_PROPAGATION_DURATION));

    float x     = x0[0];
    float y     = x0[1];
    float z     = x0[2];
    float yaw   = x0[3];
    float pitch = x0[4];
    float v     = x0[5];

    bool motionValid = true;
    float bbMin[W_DIM], bbMax[W_DIM];

    for(int i = 0; i < propagationDuration; i++)
        {
            float x0State[W_DIM] = {x, y, z};

            // --- State Propagation using 4th Order Runge-Kutta Method ---
            x +=
              (STEP_SIZE / 6.0f) *
              (v * cosf(pitch) * cosf(yaw) +
               2.0f * ((v + 0.5f * STEP_SIZE * a) * cosf(pitch + 0.5f * STEP_SIZE * pitchRate) * cosf(yaw + 0.5f * STEP_SIZE * yawRate) +
                       (v + 0.5f * STEP_SIZE * a) * cosf(pitch + 0.5f * STEP_SIZE * pitchRate) * cosf(yaw + 0.5f * STEP_SIZE * yawRate)) +
               (v + STEP_SIZE * a) * cosf(pitch + STEP_SIZE * pitchRate) * cosf(yaw + STEP_SIZE * yawRate));
            y +=
              (STEP_SIZE / 6.0f) *
              (v * cosf(pitch) * sinf(yaw) +
               2.0f * ((v + 0.5f * STEP_SIZE * a) * cosf(pitch + 0.5f * STEP_SIZE * pitchRate) * sinf(yaw + 0.5f * STEP_SIZE * yawRate) +
                       (v + 0.5f * STEP_SIZE * a) * cosf(pitch + 0.5f * STEP_SIZE * pitchRate) * sinf(yaw + 0.5f * STEP_SIZE * yawRate)) +
               (v + STEP_SIZE * a) * cosf(pitch + STEP_SIZE * pitchRate) * sinf(yaw + STEP_SIZE * yawRate));
            z += (STEP_SIZE / 6.0f) * (v * sinf(pitch) +
                                       2.0f * ((v + 0.5f * STEP_SIZE * a) * sinf(pitch + 0.5f * STEP_SIZE * pitchRate) +
                                               (v + 0.5f * STEP_SIZE * a) * sinf(pitch + 0.5f * STEP_SIZE * pitchRate)) +
                                       (v + STEP_SIZE * a) * sinf(pitch + STEP_SIZE * pitchRate));
            yaw += STEP_SIZE * yawRate;
            pitch += STEP_SIZE * pitchRate;
            v += (STEP_SIZE / 6.0f) * (a + 2.0f * (a + a) + a);

            // --- Dynamics Validity Check ---'
            if(v < V_MIN || v > V_MAX)
                {
                    motionValid = false;
                    break;
                }
            else if(pitch < DUBINS_AIRPLANE_MIN_PITCH || pitch > DUBINS_AIRPLANE_MAX_PITCH)
                {
                    motionValid = false;
                    break;
                }

            float x1State[W_DIM] = {x, y, z};

            // --- Workspace Limit Check ---
            if(x < 0 || x > W_SIZE || y < 0 || y > W_SIZE || z < 0 || z > W_SIZE)
                {
                    motionValid = false;
                    break;
                }

            // --- Obstacle Collision Check ---
            for(int d = 0; d < W_DIM; d++)
                {
                    if(x0State[d] > x1State[d])
                        {
                            bbMin[d] = x1State[d];
                            bbMax[d] = x0State[d];
                        }
                    else
                        {
                            bbMin[d] = x0State[d];
                            bbMax[d] = x1State[d];
                        }
                }

            motionValid = motionValid && isMotionValid(x0State, x1State, bbMin, bbMax, obstacles, obstaclesCount);
            if(!motionValid) break;
        }

    x1[0] = x;
    x1[1] = y;
    x1[2] = z;
    x1[3] = yaw;
    x1[4] = pitch;
    x1[5] = v;
    x1[6] = yawRate;
    x1[7] = pitchRate;
    x1[8] = a;
    x1[9] = STEP_SIZE * propagationDuration;

    return motionValid;
}

/***************************/
/* QUAD PROPAGATION FUNCTION */
/***************************/
__device__ bool propagateAndCheckQuadRungeKutta(float* x0, float* x1, curandState* seed, float* obstacles, int obstaclesCount)
{
    float Zc = QUAD_MIN_Zc + curand_uniform(seed) * (QUAD_MAX_Zc - QUAD_MIN_Zc);
    float Lc = QUAD_MIN_Lc + curand_uniform(seed) * (QUAD_MAX_Lc - QUAD_MIN_Lc);
    float Mc = QUAD_MIN_Mc + curand_uniform(seed) * (QUAD_MAX_Mc - QUAD_MIN_Mc);
    float Nc = QUAD_MIN_Nc + curand_uniform(seed) * (QUAD_MAX_Nc - QUAD_MIN_Nc);

    int propagationDuration = 1 + (int)(curand_uniform(seed) * (MAX_PROPAGATION_DURATION));

    bool motionValid = true;
    float bbMin[W_DIM], bbMax[W_DIM];

    float h0[STATE_DIM];
    float h1[STATE_DIM];
    float h2[STATE_DIM];
    float h3[STATE_DIM];
    float h4[STATE_DIM];

    for(int j = 0; j < STATE_DIM; j++) h0[j] = x0[j];

    for(int i = 0; i < propagationDuration; i++)
        {
            float x0State[W_DIM] = {h0[0], h0[1], h0[2]};

            ode(h1, h0, nullptr, Zc, Lc, Mc, Nc, 0);
            ode(h2, h0, h1, Zc, Lc, Mc, Nc, 1);
            ode(h3, h0, h2, Zc, Lc, Mc, Nc, 2);
            ode(h4, h0, h3, Zc, Lc, Mc, Nc, 3);
            for(int j = 0; j < STATE_DIM; j++)
                {
                    h0[j] += STEP_SIZE / 6.0f * (h1[j] + 2.0f * h2[j] + 2.0f * h3[j] + h4[j]);
                }

            float x1State[W_DIM] = {h0[0], h0[1], h0[2]};

            // --- Vehicle Dynamics Check ---
            if(h0[6] < V_MIN || h0[6] > V_MAX || h0[7] < V_MIN || h0[7] > V_MAX || h0[8] < V_MIN || h0[8] > V_MAX)
                {
                    motionValid = false;
                    break;
                }

            // --- Workspace Limit Check ---
            if(h0[0] < W_MIN || h0[0] > W_MAX || h0[1] < W_MIN || h0[1] > W_MAX || h0[2] < W_MIN || h0[2] > W_MAX)
                {
                    motionValid = false;
                    break;
                }

            // --- Obstacle Collision Check ---
            for(int d = 0; d < W_DIM; d++)
                {
                    if(x0State[d] > x1State[d])
                        {
                            bbMin[d] = x1State[d];
                            bbMax[d] = x0State[d];
                        }
                    else
                        {
                            bbMin[d] = x0State[d];
                            bbMax[d] = x1State[d];
                        }
                }

            motionValid = motionValid && isMotionValid(x0State, x1State, bbMin, bbMax, obstacles, obstaclesCount);
            if(!motionValid) break;
        }

    for(int j = 0; j < STATE_DIM; j++) x1[j] = h0[j];

    x1[12] = Zc;
    x1[13] = Lc;
    x1[14] = Mc;
    x1[15] = Nc;
    x1[16] = STEP_SIZE * propagationDuration;

    return motionValid;
}

__device__ void ode(float* x0dot, float* x0, float* h, float Zc, float Lc, float Mc, float Nc, int itr)
{
    float phi, theta, psi, u, v, w, p, q, r, delta;

    if(itr == 0)
        {
            delta = 1;
        }
    else if(itr == 1 || itr == 2)
        {
            delta = 0.5f * STEP_SIZE;
        }
    else if(itr == 3)
        {
            delta = STEP_SIZE;
        }

    phi   = x0[3] + delta * (h ? h[3] : 0);
    theta = x0[4] + delta * (h ? h[4] : 0);
    psi   = x0[5] + delta * (h ? h[5] : 0);
    u     = x0[6] + delta * (h ? h[6] : 0);
    v     = x0[7] + delta * (h ? h[7] : 0);
    w     = x0[8] + delta * (h ? h[8] : 0);
    p     = x0[9] + delta * (h ? h[9] : 0);
    q     = x0[10] + delta * (h ? h[10] : 0);
    r     = x0[11] + delta * (h ? h[11] : 0);

    x0dot[0] = cos(theta) * cos(psi) * u + (sin(phi) * sin(theta) * cos(psi) - cos(phi) * sin(psi)) * v +
               (cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi)) * w;
    x0dot[1] = cos(theta) * sin(psi) * u + (sin(phi) * sin(theta) * sin(psi) + cos(phi) * cos(psi)) * v +
               (cos(phi) * sin(theta) * sin(psi) - sin(phi) * cos(psi)) * w;
    x0dot[2] = -sin(theta) * u + sin(phi) * cos(theta) * v + cos(phi) * cos(theta) * w;
    x0dot[3] = p + (q * sin(phi) + r * cos(phi)) * tan(theta);
    x0dot[4] = q * cos(phi) - r * sin(phi);
    x0dot[5] = (q * sin(phi) + r * cos(phi)) / cos(theta);

    float XYZ = -NU * sqrt(u * u + v * v + w * w);
    x0dot[6]  = (r * v - q * w) - GRAVITY * sin(theta) + MASS_INV * XYZ * u;
    x0dot[7]  = (p * w - r * u) + GRAVITY * cos(theta) * sin(phi) + MASS_INV * XYZ * v;
    x0dot[8]  = (q * u - p * v) + GRAVITY * cos(theta) * cos(phi) + MASS_INV * XYZ * w + MASS_INV * Zc;

    float LMN = -MU * sqrt(p * p + q * q + r * r);
    x0dot[9]  = (IY - IZ) / IX * q * r + (1 / IX) * LMN * p + (1 / IX) * Lc;
    x0dot[10] = (IZ - IX) / IY * p * r + (1 / IY) * LMN * q + (1 / IY) * Mc;
    x0dot[11] = (IX - IY) / IZ * p * q + (1 / IZ) * LMN * r + (1 / IZ) * Nc;
}

/***************************/
/* 2D CLOHESSY-WILTSHIRE PROPAGATION FUNCTION */
/***************************/
// State x = [x_radial, y_intrack, vx, vy]. Control = one impulsive delta-V (dv_r, dv_i)
// applied at the start of the edge, then a coast on the linear CW dynamics for the edge
// duration. The CW state-transition matrix is exact, so STEP_SIZE only sets the
// collision-check / safety-integral sampling resolution (not dynamics accuracy).
// Child sample layout: x1 = [x, y, vx, vy, dv_r, dv_i, safetyPenalty, dt].
__device__ bool propagateAndCheckCW(float* x0, float* x1, curandState* seed, float* obstacles, int obstaclesCount)
{
    const float n = MEAN_MOTION;

    // --- Sample impulsive delta-V (radial, in-track) and the coast duration ---
    float dv_r              = DV_MIN + curand_uniform(seed) * (DV_MAX - DV_MIN);
    float dv_i              = DV_MIN + curand_uniform(seed) * (DV_MAX - DV_MIN);
    int propagationDuration = 1 + (int)(curand_uniform(seed) * (MAX_PROPAGATION_DURATION));

    // --- Apply the impulse once, at the start of the edge; then coast ---
    float x  = x0[0];
    float y  = x0[1];
    float vx = x0[2] + dv_r;
    float vy = x0[3] + dv_i;

    // --- CW state-transition coefficients for one STEP_SIZE coast (constant, hoisted) ---
    const float s   = sinf(n * STEP_SIZE);
    const float c   = cosf(n * STEP_SIZE);
    const float nDt = n * STEP_SIZE;

    float safetyPenalty = 0.0f;
    bool motionValid    = true;
    float bbMin[W_DIM], bbMax[W_DIM];

    for(int i = 0; i < propagationDuration; i++)
        {
            float x0State[W_DIM] = {x, y};

            // --- Exact CW propagation over one STEP_SIZE (impulse already applied) ---
            float xNew  = (4.0f - 3.0f * c) * x + (s / n) * vx + (2.0f / n) * (1.0f - c) * vy;
            float yNew  = 6.0f * (s - nDt) * x + y - (2.0f / n) * (1.0f - c) * vx + (1.0f / n) * (4.0f * s - 3.0f * nDt) * vy;
            float vxNew = 3.0f * n * s * x + c * vx + 2.0f * s * vy;
            float vyNew = -6.0f * n * (1.0f - c) * x - 2.0f * s * vx + (4.0f * c - 3.0f) * vy;
            x = xNew, y = yNew, vx = vxNew, vy = vyNew;

            // --- Velocity (dynamics) validity ---
            if(vx < V_MIN || vx > V_MAX || vy < V_MIN || vy > V_MAX)
                {
                    motionValid = false;
                    break;
                }

            float x1State[W_DIM] = {x, y};

            // --- Workspace limit check (signed, flag-centered) ---
            if(x < W_MIN || x > W_MAX || y < W_MIN || y > W_MAX)
                {
                    motionValid = false;
                    break;
                }

            // --- Obstacle collision check (hard keep-out box) ---
            for(int d = 0; d < W_DIM; d++)
                {
                    if(x0State[d] > x1State[d])
                        {
                            bbMin[d] = x1State[d];
                            bbMax[d] = x0State[d];
                        }
                    else
                        {
                            bbMin[d] = x0State[d];
                            bbMax[d] = x1State[d];
                        }
                }

            motionValid = motionValid && isMotionValid(x0State, x1State, bbMin, bbMax, obstacles, obstaclesCount);
            if(!motionValid) break;

            // --- Safety cost: integral of sum_defender (STEP_SIZE / dist_to_center) along the coast.
            // Distance is to each obstacle-box CENTER (always >= keep-out radius, since the box itself
            // is a hard no-go), so 1/dist stays bounded. ---
            for(int o = 0; o < obstaclesCount; o++)
                {
                    float cx   = 0.5f * (obstacles[o * 2 * W_DIM + 0] + obstacles[o * 2 * W_DIM + W_DIM + 0]);
                    float cy   = 0.5f * (obstacles[o * 2 * W_DIM + 1] + obstacles[o * 2 * W_DIM + W_DIM + 1]);
                    float dx   = x - cx;
                    float dy   = y - cy;
                    float dist = sqrtf(dx * dx + dy * dy);
                    if(dist < 1.0f) dist = 1.0f;  // guard (should not trigger: center sits inside the hard box)
                    safetyPenalty += STEP_SIZE / dist;
                }
        }

    x1[0] = x, x1[1] = y, x1[2] = vx, x1[3] = vy;
    x1[4] = dv_r, x1[5] = dv_i, x1[6] = safetyPenalty, x1[7] = STEP_SIZE * propagationDuration;
    return motionValid;
}

/***************************/
/* GET PROPAGATION FUNCTION */
/***************************/
__device__ PropagateAndCheckFunc getPropagateAndCheckFunc()
{
    switch(MODEL)
        {
            case 0:
                return propagateAndCheckUnicycle;
            case 1:
                return propagateAndCheckDoubleIntRungeKutta;
            case 2:
                return propagateAndCheckDubinsAirplaneRungeKutta;
            case 3:
                return propagateAndCheckQuadRungeKutta;
            case 4:
                return propagateAndCheckCW;
            default:
                return nullptr;
        }
}