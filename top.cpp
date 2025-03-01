#include "dcl.h"

void compute_attention_on_q_row(fixed_t Q_local_row[dk], fixed_t K_local[N][dk], fixed_t attention[N]) {
#pragma HLS array_partition variable=Q_local_row dim=1 complete
#pragma HLS array_partition variable=K_local dim=2 complete
    const ap_fixed<16, 8> scale = 1.0 / sqrt((float)dk);
    for (int j = 0; j < N; ++j) {
        // ap_fixed<16, 8> sum = 0;
        fixed_t sum_local[dk]; // made this to solve DSP utilization issues in this unroll
        #pragma HLS array_partition variable=sum_local dim=1 complete
        for (int k = 0; k < dk; ++k) {
        #pragma HLS unroll
            sum_local[k] = Q_local_row[k] * K_local[j][k]; 
        }

        ap_fixed<16, 8> sum = 0;
        for(int k =0; k < dk; k++) {
            #pragma HLS pipeline
            sum+=sum_local[k];
        }

        attention[j] = sum * scale;
    }
}

void compute_softmax_on_row(fixed_t attention[N]) {
#pragma HLS array_partition variable=attention dim=1 complete
    ap_fixed<32, 8> max_val = attention[0];
    for (int j = 1; j < N; ++j) {
        if (attention[j] > max_val) {
            max_val = attention[j];
        }
    }

    ap_fixed<32, 8> sum = 0;
    for (int j = 0; j < N; ++j) {
        attention[j] = hls::exp(attention[j] - max_val);
        sum += attention[j];
    }

    for (int j = 0; j < N; ++j) {
    #pragma HLS unroll
        attention[j] /= sum;
    }
}

void compute_attention_v_vec_mul_on_row(fixed_t attention[N], fixed_t V_local[N][dv], fixed_t output_vec_local[dv]) {
    for (int j = 0; j < dv; ++j) {
        ap_fixed<32, 8> sum = 0;
        for (int k = 0; k < N; ++k) {
            sum += attention[k] * V_local[k][j];
        }
        output_vec_local[j] = sum;
    }
}
// saves holding of attention mat in BRAM
void handle_row_operations(fixed_t Q_local[N][dk], fixed_t K_local[N][dk], fixed_t V_local[N][dv], fixed_t output_matrix_local[N][dv])
{
    for(int i=0; i<N; i++) {
        fixed_t attention[N];
        compute_attention_on_q_row(Q_local[i], K_local, attention);
        compute_softmax_on_row(attention);
        compute_attention_v_vec_mul_on_row(attention, V_local, output_matrix_local[i]);
    }
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
                Q_local[j][k] = Q[i][j][k];
                K_local[j][k] = K[i][j][k];
            }

            for(size_t kk = 0; kk < dv; kk++)
            {
                V_local[j][kk] = V[i][j][kk];
            }
        }

        handle_row_operations(Q_local, K_local, V_local, output_local);
        for(size_t j = 0; j < N; j++)
        {
            for(size_t kk = 0; kk < dv; kk++)
            {
                Output[i][j][kk] = output_local[j][kk];
            }
        }
    }
}


// #include "dcl.h"

// void softmax_HLS(fixed_t matrix[B][N][N]) {
//     for (int b = 0; b < B; ++b) {
//         for (int i = 0; i < N; ++i) {
//             ap_fixed<32, 8> max_val = matrix[b][i][0];
//             for (int j = 1; j < N; ++j) {
//                 if (matrix[b][i][j] > max_val) {
//                     max_val = matrix[b][i][j];
//                 }
//             }

//             ap_fixed<32, 8> sum = 0;
//             for (int j = 0; j < N; ++j) {
//                 matrix[b][i][j] = hls::exp(matrix[b][i][j] - max_val);
//                 sum += matrix[b][i][j];
//             }

//             for (int j = 0; j < N; ++j) {
//                 matrix[b][i][j] /= sum;
//             }
//         }
//     }
// }

// void compute_attention_HLS(fixed_t Q[B][N][dk], fixed_t K[B][N][dk], fixed_t V[B][N][dv], fixed_t Output[B][N][dv]) {
// #pragma HLS interface m_axi port=Q offset=slave bundle=mem1
// #pragma HLS interface m_axi port=K offset=slave bundle=mem1
// #pragma HLS interface m_axi port=V offset=slave bundle=mem1
// #pragma HLS interface m_axi port=Output offset=slave bundle=mem2

// #pragma HLS interface s_axilite port=return


//     fixed_t attention[B][N][N];
//     ap_fixed<32, 8> scale = 1.0 / sqrt((float)dk);

//     // Compute Q * K^T
//     for (int b = 0; b < B; ++b) {
//         for (int i = 0; i < N; ++i) {
//             for (int j = 0; j < N; ++j) {
//                 ap_fixed<32, 8> sum = 0;
//                 for (int k = 0; k < dk; ++k) {
//                     sum += Q[b][i][k] * K[b][j][k];
//                 }
//                 attention[b][i][j] = sum * scale;
//             }
//         }
//     }

//     // Apply softmax
//     softmax_HLS(attention);

//     // Compute Attention * V
//     for (int b = 0; b < B; ++b) {
//         for (int i = 0; i < N; ++i) {
//             for (int j = 0; j < dv; ++j) {
//                 ap_fixed<32, 8> sum = 0;
//                 for (int k = 0; k < N; ++k) {
//                     sum += attention[b][i][k] * V[b][k][j];
//                 }
//                 Output[b][i][j] = sum;
//             }
//         }
//     }

// }