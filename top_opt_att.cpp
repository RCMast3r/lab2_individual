#include "dcl.h"
#include "hls_task.h"
#include "hls_stream.h"

#define num_inner_parallel 32


void handle_vector_scale(ap_fixed<32, 8> sums[N], fixed_t attention_out[N], ap_fixed<32, 8> scale) {
#pragma HLS inline
    
    // #pragma HLS array_partition variable=sums dim=1 complete
    // #pragma HLS array_partition variable=attention_out dim=1 complete
    for(int i=0; i<N; i++) {

    #pragma HLS pipeline off
        attention_out[i] = sums[i] * scale;
    }
}

void scaled_dot_product(fixed_t qs_in[N][dk], fixed_t ks_in[N][dk], fixed_t att_out[N][N]) {
#pragma HLS inline
#define div_factor 50

    ap_fixed<32, 8> sums_local[N][N];
#pragma HLS array_partition variable=sums_local dim=1 factor=div_factor cyclic 
#pragma HLS array_partition variable=qs_in dim=1 factor=div_factor cyclic
#pragma HLS array_partition variable=att_out dim=1 factor=div_factor cyclic

    ap_fixed<32, 8> scale = 1.0 / sqrt((float)dk);
    scaled_dot_product_outer_loop: for (int i = 0; i < N; i+=div_factor) {
        #pragma HLS pipeline
        for(int ii=0; ii<div_factor; ii++){
        #pragma HLS unroll
            for (int j = 0; j < N; ++j) {
                ap_fixed<32, 8> sum = 0;
                for (int k = 0; k < dk; ++k) {
                    sum += qs_in[i+ii][k] * ks_in[j][k];
                }
                sums_local[i+ii][j] = sum;
            }
        }
    }
    UNROLL_FULL_VEC: for (int i =0; i < N; ++i){
    #pragma HLS pipeline off
        handle_vector_scale(sums_local[i], att_out[i], scale);
    }
}

void softmax_HLS(fixed_t matrix[N][N]) {
    // TP: good possibility for task-level parallelization: stream results into one-another in this for-loop
    for (int i = 0; i < N; ++i) {
    #pragma HLS pipeline off
        ap_fixed<32, 8> max_val = matrix[i][0];
        ap_fixed<32, 8> sum = 0;
        for (int j = 0; j < N; ++j) {
            matrix[i][j] = hls::exp(matrix[i][j] - max_val);
            sum += matrix[i][j];
        }
        for (int j = 0; j < N; ++j) {
            matrix[i][j] /= sum;
        }
    }
}

void output_attention(fixed_t attention_in[N][N], fixed_t V_in[N][dv], fixed_t attention_out[N][dv])
{
    for (int i = 0; i < N; ++i) {
        #pragma HLS pipeline off
        for (int j = 0; j < dv; ++j) {
            ap_fixed<32, 8> sum = 0;
            for (int k = 0; k < N; ++k) {
                sum += attention_in[i][k] * V_in[k][j];
            }
            attention_out[i][j] = sum;
        }
    }
}

// void compute_attention_HLS(MEM_TYPE Q[q_k_arr_size], MEM_TYPE K[q_k_arr_size], MEM_TYPE V[out_v_arr_size], MEM_TYPE Output[out_v_arr_size]) {
void compute_attention_HLS(MEM_TYPE Q[q_k_arr_size], MEM_TYPE K[q_k_arr_size], MEM_TYPE V[out_v_arr_size], fixed_t Output[B][N][dv]) {

#pragma HLS interface m_axi port=Q offset=slave bundle=mem1
#pragma HLS interface m_axi port=K offset=slave bundle=mem1
#pragma HLS interface m_axi port=V offset=slave bundle=mem1
#pragma HLS interface m_axi port=Output offset=slave bundle=mem2
#pragma HLS interface s_axilite port=return

    // for output indexing
    size_t output_idx=0;
    BIG_BATCH_OP: for (size_t i = 0; i < B; i++) {
        fixed_t Q_local[N][dk] = {}; // re-org into local 2D array
        fixed_t K_local[N][dk] = {}; // re-org into local 2D array
        fixed_t V_local[N][dv] = {}; // re-org into local 2D array

        fixed_t output_local[N][dv] = {};
        // Load Q into Q_local
        Q_LOAD: for (size_t j = 0; j < (N * dk) / mem_scale; j++) {
            #pragma HLS pipeline II=1
            MEM_TYPE temp = Q[i * ((N * dk) / mem_scale) + j];

            // Unpack MEM_TYPE into Q_local
            UNPACK_Q: for (size_t k = 0; k < mem_scale; k++) {
                #pragma HLS unroll
                size_t idx = j * mem_scale + k;
                if (idx < N * dk) {
                    Q_local[idx / dk][idx % dk] = temp.range((k + 1) * fixed_t_bit_length - 1, k * fixed_t_bit_length);
                }
            }
        }

        // Load K into K_local
        K_LOAD: for (size_t j = 0; j < (N * dk) / mem_scale; j++) {
            #pragma HLS pipeline II=1
            MEM_TYPE temp = K[i * ((N * dk) / mem_scale) + j];

            // Unpack MEM_TYPE into K_local
            UNPACK_K: for (size_t k = 0; k < mem_scale; k++) {
                #pragma HLS unroll
                size_t idx = j * mem_scale + k;
                if (idx < N * dk) {
                    K_local[idx / dk][idx % dk] = temp.range((k + 1) * fixed_t_bit_length - 1, k * fixed_t_bit_length);
                }
            }
        }

        // Load V into V_local
        V_LOAD: for (size_t j = 0; j < (N * dv) / mem_scale; j++) {
            #pragma HLS pipeline II=1
            MEM_TYPE temp = V[i * ((N * dv) / mem_scale) + j];

            // Unpack MEM_TYPE into V_local
            UNPACK_V: for (size_t k = 0; k < mem_scale; k++) {
                #pragma HLS unroll
                size_t idx = j * mem_scale + k;
                if (idx < N * dv) {
                    V_local[idx / dv][idx % dv] = temp.range((k + 1) * fixed_t_bit_length - 1, k * fixed_t_bit_length);
                }
            }
        }
        
        fixed_t attention[N][N];
        
        scaled_dot_product(Q_local, K_local, attention);
        softmax_HLS(attention);
        output_attention(attention, V_local, output_local);

        
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < dv; ++k) {
                Output[i][j][k] = output_local[j][k];
            }
        }

        // Q_WRITE: for(size_t j = 0; j < N; j++) {
        // #pragma HLS pipeline II=1
        //     for(size_t k=0; k < dv; k+=mem_scale) {
        //         MEM_TYPE packed_val = 0;
        //         for(size_t m=0; m < mem_scale; m++) {
        //         #pragma HLS unroll
        //             if( (k+m) < dv)
        //             {
        //                 packed_val.range((m+1) * fixed_t_bit_length - 1, m * fixed_t_bit_length) = output_local[j][k+m];
        //             }
        //         }
        //         Output[output_idx++]=packed_val;
        //     }
        // }

        // Store the result back to Output if needed
    }
}
