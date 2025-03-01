#include "dcl.h"

void compute_attention_on_batch(fixed_t Q_local[N][dk], fixed_t K_local[N][dk], fixed_t attention[N][N]) {
    ap_fixed<32, 8> scale = 1.0 / sqrt((float)dk);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            ap_fixed<32, 8> sum = 0;
            for (int k = 0; k < dk; ++k) {
            // #pragma HLS unroll
                sum += Q_local[i][k] * K_local[j][k];
            }
            attention[i][j] = sum * scale;
        }
    }
}

void compute_softmax_on_batch(fixed_t attention[N][N]) {
// #pragma HLS array_partition variable=attention complete dim=2 // do I need to do this?
    for (int i = 0; i < N; ++i) {
        ap_fixed<32, 8> max_val = attention[i][0];
        for (int j = 1; j < N; ++j) {
            if (attention[i][j] > max_val) {
                max_val = attention[i][j];
            }
        }

        for (int j = 0; j < N; ++j) {
            attention[i][j] = hls::exp(attention[i][j] - max_val);
        }

        ap_fixed<32, 8> sum = 0;
        for (int j=0; j<N; j++) {
        // #pragma HLS unroll
            sum += attention[i][j];
        }

        for (int j = 0; j < N; ++j) {
        // #pragma HLS unroll
            attention[i][j] /= sum;
        }
    }
}

void compute_attention_v_vec_mul(fixed_t attention[N][N], fixed_t V_local[N][dv], fixed_t output_matrix_local[N][dv])
{
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < dv; ++j) {
            ap_fixed<32, 8> sum = 0;
            for (int k = 0; k < N; ++k) {
                sum += attention[i][k] * V_local[k][j];
            }
            output_matrix_local[i][j] = sum;
        }
    }
}

void handle_batch_operation(fixed_t Q_local[N][dk], fixed_t K_local[N][dk], fixed_t V_local[N][dv], fixed_t output_matrix_local[N][dv])
{
    fixed_t attention[N][N];
    compute_attention_on_batch(Q_local, K_local, attention); // Q * K^T
    compute_softmax_on_batch(attention);
    compute_attention_v_vec_mul(attention, V_local, output_matrix_local);
}

void compute_attention_HLS(fixed_t Q[B][N][dk], fixed_t K[B][N][dk], fixed_t V[B][N][dv], fixed_t Output[B][N][dv]) {
#pragma HLS interface m_axi port=Q offset=slave bundle=mem1
#pragma HLS interface m_axi port=K offset=slave bundle=mem1
#pragma HLS interface m_axi port=V offset=slave bundle=mem1
#pragma HLS interface m_axi port=Output offset=slave bundle=mem2

#pragma HLS interface s_axilite port=return

    // first-things-first, we need to copy the args to BRAM from DRAM:
    // next, allocate our local output that we will copy to the back out
    for(size_t i = 0; i < B; i++)
    {
        fixed_t Q_local[N][dk];
        fixed_t K_local[N][dk];
        fixed_t V_local[N][dv];

        fixed_t output_local[N][dv];
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

        handle_batch_operation(Q_local, K_local, V_local, output_local);
        
        for(size_t j = 0; j < N; j++)
        {
            for(size_t kk = 0; kk < dv; kk++)
            {
                Output[i][j][kk] = output_local[i][j][kk];
            }
        }
    }
}