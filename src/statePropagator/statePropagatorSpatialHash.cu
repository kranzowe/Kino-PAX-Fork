#include "statePropagator/statePropagatorSpatialHash.cuh"

__device__ bool propagateAndCheckSpatialHash(float* x0, float* x1, curandState* seed, SpatialHashGrid grid, float* obstacles, int obstaclesCount)
{
    PropagateAndCheckFuncSpatialHash func = getPropagateAndCheckFuncSpatialHash();
    return func ? func(x0, x1, seed, grid, obstacles, obstaclesCount) : false;
}

/***************************/
/* UNICYCLE PROPAGATION FUNCTION WITH SPATIAL HASH */
/***************************/
__device__ bool propagateAndCheckUnicycleSpatialHash(float* x0, float* x1, curandState* seed, SpatialHashGrid grid, float* obstacles, int obstaclesCount)
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

        x += v * cosTheta * STEP_SIZE;
        y += v * sinTheta * STEP_SIZE;
        theta += (v / UNI_LENGTH) * tanSteering * STEP_SIZE;
        v += a * STEP_SIZE;
        float x1State[W_DIM] = {x, y};

        if(x < 0 || x > W_SIZE || y < 0 || y > W_SIZE)
        {
            motionValid = false;
            break;
        }

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

        motionValid = motionValid && isMotionValidSpatialHash(x0State, x1State, bbMin, bbMax, grid, obstacles);
        if(!motionValid) break;
    }

    x1[0] = x, x1[1] = y, x1[2] = theta, x1[3] = v, x1[4] = a, x1[5] = steering, x1[6] = duration;
    return motionValid;
}

/***************************/
/* DOUBLE INTEGRATOR PROPAGATION FUNCTION WITH SPATIAL HASH */
/***************************/
__device__ bool propagateAndCheckDoubleIntRungeKuttaSpatialHash(float* x0, float* x1, curandState* seed, SpatialHashGrid grid, float* obstacles, int obstaclesCount)
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

        x += (vx + (vx + 2 * (vx + ax * STEP_SIZE / 2) + (vx + ax * STEP_SIZE))) * STEP_SIZE / 6;
        y += (vy + (vy + 2 * (vy + ay * STEP_SIZE / 2) + (vy + ay * STEP_SIZE))) * STEP_SIZE / 6;
        z += (vz + (vz + 2 * (vz + az * STEP_SIZE / 2) + (vz + az * STEP_SIZE))) * STEP_SIZE / 6;
        vx += (ax + 2 * ax + 2 * ax + ax) * STEP_SIZE / 6;
        vy += (ay + 2 * ay + 2 * ay + ay) * STEP_SIZE / 6;
        vz += (az + 2 * az + 2 * az + az) * STEP_SIZE / 6;

        if(vx < V_MIN || vx > V_MAX || vy < V_MIN || vy > V_MAX || vz < V_MIN || vz > V_MAX)
        {
            motionValid = false;
            break;
        }

        float x1State[W_DIM] = {x, y, z};

        if(x < 0 || x > W_SIZE || y < 0 || y > W_SIZE || z < 0 || z > W_SIZE)
        {
            motionValid = false;
            break;
        }

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

        motionValid = motionValid && isMotionValidSpatialHash(x0State, x1State, bbMin, bbMax, grid, obstacles);
        if(!motionValid) break;
    }

    x1[0] = x, x1[1] = y, x1[2] = z, x1[3] = vx, x1[4] = vy, x1[5] = vz, x1[6] = ax, x1[7] = ay, x1[8] = az,
    x1[9] = STEP_SIZE * propagationDuration;
    return motionValid;
}

/***************************/
/* DUBINS AIRPLANE PROPAGATION FUNCTION WITH SPATIAL HASH */
/***************************/
__device__ bool propagateAndCheckDubinsAirplaneRungeKuttaSpatialHash(float* x0, float* x1, curandState* seed, SpatialHashGrid grid, float* obstacles, int obstaclesCount)
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

        if(x < 0 || x > W_SIZE || y < 0 || y > W_SIZE || z < 0 || z > W_SIZE)
        {
            motionValid = false;
            break;
        }

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

        motionValid = motionValid && isMotionValidSpatialHash(x0State, x1State, bbMin, bbMax, grid, obstacles);
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
/* QUAD PROPAGATION FUNCTION WITH SPATIAL HASH */
/***************************/
__device__ bool propagateAndCheckQuadRungeKuttaSpatialHash(float* x0, float* x1, curandState* seed, SpatialHashGrid grid, float* obstacles, int obstaclesCount)
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

        if(h0[6] < V_MIN || h0[6] > V_MAX || h0[7] < V_MIN || h0[7] > V_MAX || h0[8] < V_MIN || h0[8] > V_MAX)
        {
            motionValid = false;
            break;
        }

        if(h0[0] < W_MIN || h0[0] > W_MAX || h0[1] < W_MIN || h0[1] > W_MAX || h0[2] < W_MIN || h0[2] > W_MAX)
        {
            motionValid = false;
            break;
        }

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

        motionValid = motionValid && isMotionValidSpatialHash(x0State, x1State, bbMin, bbMax, grid, obstacles);
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

/***************************/
/* GET PROPAGATION FUNCTION */
/***************************/
__device__ PropagateAndCheckFuncSpatialHash getPropagateAndCheckFuncSpatialHash()
{
    switch(MODEL)
    {
        case 0:
            return propagateAndCheckUnicycleSpatialHash;
        case 1:
            return propagateAndCheckDoubleIntRungeKuttaSpatialHash;
        case 2:
            return propagateAndCheckDubinsAirplaneRungeKuttaSpatialHash;
        case 3:
            return propagateAndCheckQuadRungeKuttaSpatialHash;
        default:
            return nullptr;
    }
}
