#include <metal_stdlib>
using namespace metal;

kernel void shiftAndInsertKernel(
    device float* delayBuffer [[buffer(0)]],
    device const float* inputBuffer [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    // Insert new elements at the beginning of the delay buffer
    if (tid < size) {
        delayBuffer[tid] = inputBuffer[tid];
    }

    // Shift elements
    if (tid + size < 2 * size) {
        delayBuffer[tid + size] = delayBuffer[tid];
    }
}
