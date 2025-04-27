/*
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;

kernel void shared_partitioned_convolution(
    device atomic_float* Result [[buffer(0)]],       // Output Result array
    device float* Dry [[buffer(1)]],    // Input Dry signal
    device float* Imp [[buffer(2)]],    // Input Impulse response
    device const uint* SIZES [[buffer(3)]],   // SIZES array: SIZES[0] = length of Dry/Imp, SIZES[1] = length of Result
    threadgroup   float* partArray,
                                           //threadgroup float* partArray [[threadgroup(0)]],  // Shared memory
    uint gid [[thread_position_in_grid]], uint threadid [[thread_position_in_threadgroup]], uint blockDim [[threads_per_threadgroup]], uint blockid[[threadgroup_position_in_grid]]) {

    // Calculate thread index
    uint thread_idx  = blockid*blockDim+threadid;
    uint tid = threadid;
   // const uint copy_idx = thread_position_in_threadgroup;
    const uint block_size = SIZES[0]; // Length of Dry and Impulse arrays
    const uint result_size = SIZES[1]; // Length of Result array
    // Define pointers to shared memory sections
        threadgroup   float* shared_arr1 = &partArray[0];         // Dry signal
        threadgroup   float* shared_arr2 = &partArray[block_size];         // Impulse response
        threadgroup   float* tempResult = &partArray[2 * block_size]; // Temporary result buffer
    // Initialize shared memory
    
       
  //  threadgroup_barrier(mem_flags::mem_threadgroup);

        // Perform convolution
        
        tempResult[tid] = 0.f;
        tempResult[tid + block_size] = 0.f;
        // Load data into shared memory
        shared_arr1[tid] = Dry[thread_idx];
        shared_arr2[tid] = Imp[thread_idx];
    threadgroup_barrier(mem_flags::mem_threadgroup);
        // Perform convolution with loop unrolling and careful index handling
        
        float sum = 0.0f;
        for (uint j = 0; j <= tid; j++) {
            sum += shared_arr1[tid - j] * shared_arr2[j];
        }
        tempResult[tid] = sum;

        float sum2 = 0.f;

      //  sum2 += shared_arr1[tid] * shared_arr2[block_size - tid];
      //  tempResult[block_size] += sum2;
         

        for (uint i = tid + block_size; i < result_size; i += blockDim) {
            float sum3 = 0.0f;

            uint start_j = max(0u, i - block_size + 1u);  // Prevent negative indices
            uint end_j = block_size;

            for (uint j = start_j; j < end_j; j++) {
               uint idx1 = i - j;  // Index for shared_arr1
                if (idx1 >= 0 && idx1 < block_size) {  // Validate index
                    sum3 += shared_arr1[idx1] * shared_arr2[j];
                }
            }
            tempResult[i] = sum3;

        }
    threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint i = 0; i <result_size ; i++)
        atomic_fetch_add_explicit(&Result[i],tempResult[i],memory_order_relaxed);
     
 }
*/
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;

kernel void shared_partitioned_convolution(
    device atomic_float* Result [[buffer(0)]],       // Output Result array
    device float* Dry [[buffer(1)]],    // Input Dry signal
    device float* Imp [[buffer(2)]],    // Input Impulse response
    device const uint* SIZES [[buffer(3)]],   // SIZES array: SIZES[0] = length of Dry/Imp, SIZES[1] = length of Result
    threadgroup   float* partArray,
                                           //threadgroup float* partArray [[threadgroup(0)]],  // Shared memory
    uint gid [[thread_position_in_grid]], uint threadid [[thread_position_in_threadgroup]], uint blockDim [[threads_per_threadgroup]], uint blockid[[threadgroup_position_in_grid]]) {

    // Calculate thread index
    uint thread_idx  = blockid*blockDim+threadid;
    uint tid = threadid;
   // const uint copy_idx = thread_position_in_threadgroup;
    const uint block_size = SIZES[0]; // Length of Dry and Impulse arrays
    const uint result_size = SIZES[1]; // Length of Result array
    // Define pointers to shared memory sections
        threadgroup   float* shared_arr1 = &partArray[0];         // Dry signal
        threadgroup   float* shared_arr2 = &partArray[block_size];         // Impulse response
        threadgroup   float* tempResult = &partArray[2 * block_size]; // Temporary result buffer
   
        
        
        tempResult[tid] = 0.f;
        tempResult[tid + block_size] = 0.f;
        // Load data into shared memory
        shared_arr1[tid] = Dry[thread_idx];
        shared_arr2[tid] = Imp[thread_idx];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    
        
      
        for (uint j = 0; j < block_size; j++) {
            uint inv = (tid - j) % block_size;
            tempResult[j + inv] += (shared_arr1[j] * shared_arr2[inv]) * 0.015f;
        }
        
    threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint i = 0; i <result_size ; i++)
        atomic_fetch_add_explicit(&Result[i],tempResult[i],memory_order_relaxed);
     
 }

