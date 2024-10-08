#include <metal_stdlib>
using namespace metal;

kernel void shared_partitioned_convolution4(
    // device atomic_float* Result [[buffer(0)]],
    device float* Result [[buffer(0)]],
    device const float* Dry [[buffer(1)]],
    device const float* Imp [[buffer(2)]],
    constant uint* SIZES [[buffer(3)]],
    threadgroup float* partArray [[threadgroup(0)]],
    uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]],
    uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
    uint threads_per_threadgroup        [[ threads_per_threadgroup ]]) {
    uint thread_position_in_grid =
            (threadgroup_position_in_grid * threads_per_threadgroup) +
            thread_position_in_threadgroup;
    // Define shared memory (split into two sections: one for Dry, one for Impulse)
    threadgroup float* arr1 = &partArray[0];          // Dry signal
    threadgroup float* arr2 = &partArray[SIZES[0]];   // Impulse response
    
    
        // Copy data into shared memory (threadgroup memory)
        arr1[thread_position_in_threadgroup] = Dry[thread_position_in_grid];
        arr2[thread_position_in_threadgroup] = Imp[thread_position_in_grid];
    

    // Synchronize threads before performing the convolution
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
        // Perform convolution
            for (uint i = 0; i < SIZES[0]; i++) {
                // Calculate the inverse index for linear convolution
                uint inv = (SIZES[0] - thread_position_in_threadgroup) % SIZES[0]; // Adjust the index

                //partial convolution result
                float sum = arr1[i] * arr2[inv];

                // Atomic add to the result buffer
               // atomic_fetch_add_explicit(& Result[i + inv], sum, memory_order_relaxed);
                Result[i + inv] += sum;
                
            }
}
