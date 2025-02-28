#include "dcl.h"

using namespace std;


void pack_out_v_tensor(fixed_t input[B][N][dv], MEM_TYPE packed_arr[], int dim) {
    int idx = 0;
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < dim; k += mem_scale) {
                MEM_TYPE packed_val = 0;
                for (int m = 0; m < mem_scale; m++) {
                    if (k + m < dim) {
                        packed_val.range((m + 1) * fixed_t_bit_length - 1, m * fixed_t_bit_length) = input[i][j][k + m];
                    }
                }
                packed_arr[idx++] = packed_val;
            }
        }
    }
}

void pack_q_k_tensor(fixed_t input[B][N][dk], MEM_TYPE packed_arr[], int dim) {
    int idx = 0;
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < dim; k += mem_scale) {
                MEM_TYPE packed_val = 0;
                for (int m = 0; m < mem_scale; m++) {
                    if (k + m < dim) {
                        packed_val.range((m + 1) * fixed_t_bit_length - 1, m * fixed_t_bit_length) = input[i][j][k + m];
                    }
                }
                packed_arr[idx++] = packed_val;
            }
        }
    }
}

void unpack_tensor(MEM_TYPE packed_arr[], fixed_t output[B][N][dv], int dim) {
    int idx = 0;
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < dim; k += mem_scale) {
                MEM_TYPE packed_val = packed_arr[idx++];
                for (int m = 0; m < mem_scale; m++) {
                    if (k + m < dim) {
                        output[i][j][k + m] = packed_val.range((m + 1) * fixed_t_bit_length - 1, m * fixed_t_bit_length);
                    }
                }
            }
        }
    }
}


void load_tensor(const char* filename, fixed_t tensor[][N][dk], int D) {
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }

    size_t elements_read = fread(tensor, sizeof(fixed_t), B * N * D, file);
    if (elements_read != B * N * D) {
        fprintf(stderr, "Error reading file: %s\n", filename);
        fclose(file);
        exit(1);
    }

    fclose(file);
}


int main() {
    // Allocate memory for tensors
    fixed_t Q[B][N][dk];
    fixed_t K[B][N][dk];
    fixed_t V[B][N][dv];
    fixed_t Output_ref[B][N][dv];
	fixed_t Output_HLS[B][N][dv];

    MEM_TYPE Q_packed[q_k_arr_size];
    MEM_TYPE K_packed[q_k_arr_size];
    MEM_TYPE V_packed[out_v_arr_size];
    MEM_TYPE Output_packed[out_v_arr_size];

    // Load tensors from binary files
    load_tensor("Q_tensor.bin", Q, dk);
    load_tensor("K_tensor.bin", K, dk);
    load_tensor("V_tensor.bin", V, dv);
	load_tensor("Output_tensor.bin", Output_ref, dv);

	for(int i = 0; i < B; i++) {
		for(int j = 0; j < N; j++) {
			for(int k = 0; k < dv; k++) {
				Output_HLS[i][j][k] = 0;
			}
		}
	}

    pack_q_k_tensor(Q, Q_packed, dk);
    pack_q_k_tensor(K, K_packed, dk);
    pack_out_v_tensor(V, V_packed, dv);

    // Call the HLS kernel
    compute_attention_HLS(Q_packed, K_packed, V_packed, Output_packed);
    unpack_tensor(Output_packed, Output_HLS, dv);
	float error = 0;
	// compare HLS output and reference output tensor
	for(int i = 0; i < B; i++) {
		for(int j = 0; j < N; j++) {
			for(int k = 0; k < dv; k++) {
				error += std::pow(Output_HLS[i][j][k].to_float() - Output_ref[i][j][k].to_float(), 2);
                //printf("Output_HLS[%d][%d][%d]: %.8f; Output_ref[%d][%d][%d]: %.8f\n", 
                //i, j, k, Output_HLS[i][j][k].to_float(), i, j, k, Output_ref[i][j][k].to_float());
			}
		}
	}
    
	error = error / (B * N * dv);
	printf("MSE: %.8f\n", error);

    return 0;
}
