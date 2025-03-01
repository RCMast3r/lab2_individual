#include "dcl.h"
#include "hls_task.h"
#include "hls_stream.h"

void handle_vector_scale(ap_fixed<32, 8> sums[N], fixed_t attention_out[N], ap_fixed<32, 8> scale) {
#pragma HLS inline
    
    // #pragma HLS array_partition variable=sums dim=1 complete
    // #pragma HLS array_partition variable=attention_out dim=1 complete
    for(int i=0; i<N; i++) {
    #pragma HLS unroll
        attention_out[i] = sums[i] * scale;
    }
}

template<size_t unroll_size, size_t size>
void scale_vecs(fixed_t vec_a[size], fixed_t vec_b[size], fixed_t vec_out[size]) {
#pragma HLS array_partition variable=vec_a dim=1 factor=unroll_size cyclic
#pragma HLS array_partition variable=vec_b dim=1 factor=unroll_size cyclic
#pragma HLS array_partition variable=vec_out dim=1 factor=unroll_size cyclic

    scalar_op: for(int i=0; i< size; i+=unroll_size) {
    #pragma HLS pipeline
        for(int ii=0; ii<unroll_size; ii++) {
        #pragma HLS unroll
            vec_out[i+ii] = vec_a[i+ii] * vec_b[i+ii];
        }
    }

}

void scaled_dot_product(fixed_t qs_in[N][dk], fixed_t ks_in[N][dk], fixed_t att_out[N][N]){
#define div_factor 2

// ap_fixed<32, 8> sums_local[N][N];
// #pragma HLS array_partition variable=sums_local dim=1 factor=div_factor cyclic 
#pragma HLS array_partition variable=qs_in dim=1 factor=div_factor cyclic
#pragma HLS array_partition variable=ks_in dim=1 factor=div_factor cyclic

    ap_fixed<32, 8> scale = 1.0 / sqrt((float)dk);
    for(int i=0; i < N; i++) {
        ap_fixed<32, 8> sums_local[N];
        #pragma HLS array_partition variable=sums_local dim=1 factor=div_factor cyclic

        for (int j = 0; j < N; j+=div_factor) {
            #pragma HLS pipeline
            for(int jj=0; jj<div_factor; jj++){
            #pragma HLS unroll
            
                fixed_t vec_mul[dk];
                // allows for micro-managed scaling parallelization
                scale_vecs<dk/4, dk>(qs_in[i], ks_in[j+jj], vec_mul);
                ap_fixed<16, 8> sum = 0;

                // is this parallize-able trivially? or do i have to implement
                for (int k = 0; k < dk; ++k) {
                    sum += vec_mul[k];
                }
                sums_local[j+jj] = sum;
            }
        }
        UNROLL_FULL_VEC: for (int i =0; i < N; ++i) {
            handle_vector_scale(sums_local, att_out[i], scale);
        }
    }
}

void output_attention_within_softmax(fixed_t attention_in[N], fixed_t V_in[N][dv], fixed_t attention_out[dv]) {
#pragma HLS array_partition variable=attention_in dim=1 complete
#pragma HLS array_partition variable=V_in dim=1 complete
    for (int j = 0; j < dv; ++j) {
        ap_fixed<32, 8> to_sum_local[N];
        #pragma HLS array_partition variable=to_sum_local dim=1 complete
        for (int k = 0; k < N; k++) {
        #pragma HLS unroll
            to_sum_local[k] = attention_in[k] * V_in[k][j];
        }

        ap_fixed<32, 8> sum = 0;
        for(int kk=0; kk < N; kk++) {
            sum+=to_sum_local[kk];
        }
        attention_out[j] = sum;
    }
}

void handle_normalization(fixed_t att_matrix_ith[N], ap_fixed<32, 8> sum) {
#pragma HLS array_partition variable=att_matrix_ith dim=1 complete
    for (int j = 0; j < N; ++j) {
    #pragma HLS unroll
        att_matrix_ith[j] /= sum;
    }
}

// softmax exponential 
void softmax_hls_comb(fixed_t att_matrix[N][N], fixed_t V_in[N][dv], fixed_t attention_out[N][dv]) {
    for (int i = 0; i < N; ++i) {
        ap_fixed<32, 8> max_val = att_matrix[i][0];
        ap_fixed<32, 8> sum = 0;
        for (int j = 0; j < N; ++j) {
        #pragma HLS pipeline off // enforced here due to 
            att_matrix[i][j] = hls::exp(att_matrix[i][j] - max_val);
            sum += att_matrix[i][j];
        }

        handle_normalization(att_matrix[i], sum);

        output_attention_within_softmax(att_matrix[i], V_in, attention_out[i]);
    }
}
void softmax_HLS(fixed_t matrix[N][N]) {
// #pragma HLS inline
    // TP: good possibility for task-level parallelization: stream results into one-another in this for-loop
    for (int i = 0; i < N; ++i) {
        ap_fixed<32, 8> max_val = matrix[i][0];
        ap_fixed<32, 8> sum = 0;
        for (int j = 0; j < N; ++j) {
        #pragma HLS pipeline off // enforced here due to 
            matrix[i][j] = hls::exp(matrix[i][j] - max_val);
            sum += matrix[i][j];
        }

        for (int j = 0; j < N; ++j) {
            matrix[i][j] /= sum;
        }

    }
}



void output_attention(fixed_t attention_in[N][N], fixed_t V_in[N][dv], fixed_t attention_out[N][dv]) {
// parallelize dim 1 of V_in and dim 2 of attention in
// made a vector that will get summed in a different loop
// in effect, parallizing the vec mul
// #define div_factor_output_attention 16

// these allow for the inner unroll
// #pragma HLS array_partition variable=attention_in dim=2
// #pragma HLS array_partition variable=V_in dim=1
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < dv; ++j) {
        // for(int j=0; j<dv; j+=div_factor_output_attention){
        // #pragma HLS pipeline
        // for(int jj = 0; jj<div_factor_output_attention; jj++) {
        // #pragma HLS unroll
        // 
            ap_fixed<32, 8> to_sum_local[N];
            // #pragma HLS array_partition variable=to_sum_local dim=1
            for (int k = 0; k < N; k) {
                // #pragma HLS unroll
                to_sum_local[k] = attention_in[i][k] * V_in[k][j];
            }

            ap_fixed<32, 8> sum = 0;
            for(int kk=0; kk < N; kk++) {
                sum+=to_sum_local[kk];
            }
            attention_out[i][j] = sum;
        }
    }
}


handle_batch_computation(Q_local[i], K_local[i], V_local[i], output_local[i]);

void compute_attention_HLS(fixed_t Q[B][N][dk], fixed_t K[B][N][dk], fixed_t V[B][N][dv], fixed_t Output[B][N][dv]) {
#pragma HLS interface m_axi port=Q offset=slave bundle=mem1
#pragma HLS interface m_axi port=K offset=slave bundle=mem1
#pragma HLS interface m_axi port=V offset=slave bundle=mem1
#pragma HLS interface m_axi port=Output offset=slave bundle=mem2

#pragma HLS interface s_axilite port=return

    // first-things-first, we need to copy the args to BRAM from DRAM:
    fixed_t Q_local[B][N][dk];
    fixed_t K_local[B][N][dk];
    fixed_t V_local[B][N][dv];

    // next, allocate our local output that we will copy to the back out
    fixed_t output_local[B][N][dv];
    for(size_t i = 0; i < B; i++)
    {
        for(size_t j = 0; j < N; j++)
        {
            for(size_t k = 0; k < dk; k++)
            {
                Q_local[i][j][k] = Q[i][j][k];
                K_local[i][j][k] = K[i][j][k];
            }

            for(size_t kk = 0; kk < dv; kk++)
            {
                V_local[i][j][kk] = V[i][j][kk];
            }
        }
    }

    for(size_t i = 0; i < B; i++) {
        
        fixed_t attention[N][N];
        scaled_dot_product(Q_local[i], K_local[i], attention);
        softmax_hls_comb(attention, V_local[i], output_local[i]);

        // handle_batch_computation(Q_local[i], K_local[i], V_local[i], output_local[i]);
        // softmax_HLS(attention);
        // output_attention(attention, V_local[i], output_local[i]);
    }

// write out to the actual output from BRAM
    for(size_t i = 0; i < B; i++)
    {
        for(size_t j = 0; j < N; j++)
        {
            for(size_t kk = 0; kk < dv; kk++)
            {
                Output[i][j][kk] = output_local[i][j][kk];
            }
        }
    }
}