#include <metal_stdlib>
using namespace metal;

kernel void shiftAndInsertKernel(
    device float* delayBuffer [[buffer(0)]],
    device const float* inputBuffer [[buffer(1)]],
    constant uint* SIZES [[buffer(2)]],
    uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]],
    uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
    uint threads_per_threadgroup        [[ threads_per_threadgroup ]])
{
    
    uint thread_position_in_grid =
            (threadgroup_position_in_grid * threads_per_threadgroup) +
            thread_position_in_threadgroup;
    // Insert new elements at the beginning of the delay buffer
    if (thread_position_in_grid < SIZES[0]) {
        delayBuffer[thread_position_in_grid] = inputBuffer[thread_position_in_grid];
    }
    
            // Shift the old values
            delayBuffer[thread_position_in_grid + SIZES[0]] = delayBuffer[thread_position_in_grid];
}
