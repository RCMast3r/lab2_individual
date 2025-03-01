
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ap_fixed.h>
#include <hls_math.h>
#include <stdlib.h>
#include <cstdint>


#define fixed_t_bit_length 16
typedef ap_fixed<fixed_t_bit_length, 5> fixed_t;

// Define tensor dimensions
#define B   (4)      // Batch size
// #define N   (100)     // Sequence length
// #define dk  (128)     // Key/Query dimension
// #define dv  (128)     // Value dimension
#define N   (10)     // Sequence length
#define dk  (12)     // Key/Query dimension
#define dv  (12)     // Value dimension

#define mem_scale 4
// the memory is 72 bits wide, we will pull in 4 at a time (16*4)
#define mem_size (fixed_t_bit_length * mem_scale)
typedef ap_uint<mem_size> MEM_TYPE;

#define q_k_arr_size ( (B*N*dk) / (mem_scale) )
#define out_v_arr_size ( (B*N*dv) / (mem_scale) )

// void compute_attention_HLS(fixed_t Q[B][N][dk], fixed_t K[B][N][dk], fixed_t V[B][N][dv], fixed_t Output[B][N][dv]);

// void compute_attention_HLS(MEM_TYPE Q[q_k_arr_size], MEM_TYPE K[q_k_arr_size], MEM_TYPE V[out_v_arr_size], MEM_TYPE Output[out_v_arr_size]);
void compute_attention_HLS(MEM_TYPE Q[q_k_arr_size], MEM_TYPE K[q_k_arr_size], MEM_TYPE V[out_v_arr_size], fixed_t Output[B][N][dv]);

