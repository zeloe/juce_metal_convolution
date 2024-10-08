#include <metal_stdlib>
using namespace metal;

kernel void shared_partitioned_convolution4(
    device float* Result [[buffer(0)]],
    device const float* Dry [[buffer(1)]],
    device const float* Imp [[buffer(2)]],
    constant uint* SIZES [[buffer(3)]],
    threadgroup float* partArray [[threadgroup(0)]],
    uint thread_idx [[thread_position_in_grid]],
    uint copy_idx [[thread_position_in_threadgroup]]
) {
    // Define shared memory
    threadgroup float* arr1 = &partArray[0];
    threadgroup float* arr2 = &partArray[SIZES[0]];

    // Copy data into shared memory (threadgroup memory)
    arr1[copy_idx] = Dry[thread_idx];
    arr2[copy_idx] = Imp[thread_idx];

    // Synchronize threads
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Perform convolution
    for (int i = 0; i < SIZES[0]; i++) {
        int inv = (SIZES[0] - copy_idx) % SIZES[0];
        Result[i + inv] = arr1[i] * arr2[inv];
    }
}
