#include "dcl.h"
#include "hls_task.h"
#include "hls_stream.h"

#define num_inner_parallel 32


void scaled_dot_product(fixed_t qs_in[N][dk], fixed_t ks_in[N][dk], fixed_t att_out[N][N])
{
    ap_fixed<32, 8> scale = 1.0 / sqrt((float)dk);
    
    // i want make the outer loop "threaded" as there is no data inter-deps here
    
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            ap_fixed<32, 8> sum = 0;
            for (int k = 0; k < dk; ++k) {
                sum += qs_in[i][k] * ks_in[j][k];
            }
            att_out[i][j] = sum * scale;
        }
    }

}

void softmax_HLS(fixed_t matrix[N][N]) {

    // TP: good possibility for task-level parallelization: stream results into one-another in this for-loop
    for (int i = 0; i < N; ++i) {
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
        for (int j = 0; j < dv; ++j) {
            ap_fixed<32, 8> sum = 0;
            for (int k = 0; k < N; ++k) {
                sum += attention_in[i][k] * V_in[k][j];
            }
            attention_out[i][j] = sum;
        }
    }
}

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

    for(size_t i = 0; i < B; i++)
    {
#pragma HLS unroll
        fixed_t attention[N][N];
        scaled_dot_product(Q_local[i], K_local[i], attention);
        softmax_HLS(attention);
        output_attention(attention, V_local[i], output_local[i]);
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