AUTOPILOT_ROOT :=/tools/software/xilinx/Vitis_HLS/2023.1

ASSEMBLE_SRC_ROOT := .

IFLAG += -I "${AUTOPILOT_ROOT}/include"
IFLAG += -D__SIM_FPO__ -D__SIM_OPENCV__ -D__SIM_FFT__ -D__SIM_FIR__ -D__SIM_DDS__ -D__DSP48E1__ -DHLS_NO_XIL_FPO_LIB
IFLAG += -g 

CFLAG += -fPIC -O0
CFLAG += -lm
CFLAG += -std=c++11 -Wno-unused-result 

all:
	g++ generate_tensors.cpp -o result_gen_tens $(CFLAG) $(IFLAG)
	g++ compute_attention.cpp -o result_compute_att $(CFLAG) $(IFLAG)
	g++ host.cpp top.cpp -o result $(CFLAG) $(IFLAG)

top:
	g++ host.cpp top.cpp -o result $(CFLAG) $(IFLAG)
	
clean:
	rm -f *.o result
