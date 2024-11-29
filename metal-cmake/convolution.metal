#include <metal_stdlib>
using namespace metal;

kernel void shared_partitioned_convolution(
    device float* Result [[buffer(0)]],       // Output Result array
    device float* Dry [[buffer(1)]],    // Input Dry signal
    device float* Imp [[buffer(2)]],    // Input Impulse response
    device const uint* SIZES [[buffer(3)]],   // SIZES array: SIZES[0] = length of Dry/Imp, SIZES[1] = length of Result
    threadgroup float* partArray [[threadgroup(0)]],  // Shared memory
    uint threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]) {

    // Calculate thread index
    uint thread_idx = (threadgroup_position_in_grid * threads_per_threadgroup) + thread_position_in_threadgroup;
    uint copy_idx = thread_position_in_threadgroup;
    uint N = SIZES[0]; // Length of Dry and Impulse arrays
    uint M = SIZES[1]; // Length of Result array

    // Define pointers to shared memory sections
    threadgroup float* arr1 = &partArray[0];         // Dry signal
    threadgroup float* arr2 = &partArray[N];         // Impulse response
    threadgroup float* tempResult = &partArray[2 * N]; // Temporary result buffer

    // Initialize shared memory
    
        arr1[copy_idx] = Dry[thread_idx];
        arr2[copy_idx] = Imp[thread_idx];
        tempResult[copy_idx] = 0.0f;
        tempResult[copy_idx + N] = 0.0f;
   
    // Synchronize threads
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Perform convolution
    for (uint i = 0; i < N; i++) {
        uint inv = (i - copy_idx) % N;
        tempResult[i + inv] += arr1[i] * arr2[inv];
    }

    // Synchronize threads
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Accumulate results into global memory (only by the first thread in the group)
    if (copy_idx == 0) {
        for (uint i = 0; i < M; i++) {
            Result[i] += tempResult[i];
        }
    }
}
